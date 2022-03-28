from .utils.utils_profiling import *  # load before other local modules

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext

from typing import Dict
from .equivariant_attention.from_se3cnn import utils_steerable
from .equivariant_attention.fibers import Fiber, fiber2head
from .utils.utils_logging import log_gradient_norm


@profile
def get_basis(S, max_degree, compute_gradients):
    """Precompute the SE(3)-equivariant weight basis, W_J^lk(x)

    This is called by get_basis_and_r().

    Args:
        S: spherical signals
        max_degree: non-negative int for degree of highest feature type
        compute_gradients: boolean, whether to compute gradients during basis construction
    Returns:
        dict of equivariant bases. Keys are in the form 'd_in,d_out'. Values are
        tensors of shape (batch_size, 1, 2*d_out+1, 1, 2*d_in+1, number_of_bases)
        where the 1's will later be broadcast to the number of output and input
        channels
    """
    if compute_gradients:
        context = nullcontext()
    else:
        context = torch.no_grad()

    with context:
        cloned_d = torch.clone(S['d'])

        if S['d'].requires_grad:
            cloned_d.requires_grad_()
            log_gradient_norm(cloned_d, 'Basis computation flow')

        # Relative positional encodings (vector)
        r_ij = utils_steerable.get_spherical_from_cartesian_torch(cloned_d)
        # Spherical harmonic basis
        Y = utils_steerable.precompute_sh(r_ij, 2*max_degree)
        device = Y[0].device

        basis = {}
        for d_in in range(max_degree+1):
            for d_out in range(max_degree+1):
                K_Js = []
                for J in range(abs(d_in-d_out), d_in+d_out+1):
                    # Get spherical harmonic projection matrices
                    Q_J = utils_steerable._basis_transformation_Q_J(J, d_in, d_out)
                    Q_J = Q_J.float().T.to(device)

                    # Create kernel from spherical harmonics
                    K_J = torch.matmul(Y[J], Q_J)
                    K_Js.append(K_J)

                # Reshape so can take linear combinations with a dot product
                size = (-1, 1, 2*d_out+1, 1, 2*d_in+1, 2*min(d_in, d_out)+1)
                basis[f'{d_in},{d_out}'] = torch.stack(K_Js, -1).view(*size)
        return basis


def get_r(S):
    """Compute internodal distances"""
    cloned_d = torch.clone(S['d'])

    if S['d'].requires_grad:
        cloned_d.requires_grad_()
        log_gradient_norm(cloned_d, 'Neural networks flow')

    # Relative positional encodings (vector)

    r = utils_steerable.get_spherical_from_cartesian_torch(cloned_d)

    return torch.sqrt(torch.sum(r**2, -1, keepdim=True))


def get_basis_and_r(S, max_degree, compute_gradients=False):
    """Return equivariant weight basis (basis) and internodal distances (r).

    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function
    can be shared as input across all SE(3)-Transformer layers in a model.

    Args:
        G: DGL graph instance of type dgl.DGLGraph()
        max_degree: non-negative int for degree of highest feature-type
        compute_gradients: controls whether to compute gradients during basis construction
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
        vector of relative distances, ordered according to edge ordering of G
    """
    basis = get_basis(S, max_degree, compute_gradients)
    r = get_r(S)
    return basis, r


### SO(3) equivariant operations on spherical signals

class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""
    def __init__(self, num_freq, in_dim, out_dim):
        """NN parameterized radial profile function.

        Args:
            num_freq: number of output frequencies
            in_dim: multiplicity of input (num input channels)
            out_dim: multiplicity of output (num output channels)
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = 32
        self.out_dim = out_dim

        self.net = nn.Sequential(nn.Linear(1,self.mid_dim),
                                 BN(self.mid_dim),
                                 nn.ReLU(),
                                 #nn.SELU(),
                                 nn.Linear(self.mid_dim,self.mid_dim),
                                 BN(self.mid_dim),
                                 nn.ReLU(),
                                 #nn.SELU(),
                                 nn.Linear(self.mid_dim,self.num_freq*in_dim*out_dim))

        nn.init.kaiming_uniform_(self.net[0].weight)
        nn.init.kaiming_uniform_(self.net[3].weight)
        nn.init.kaiming_uniform_(self.net[6].weight)

    def __repr__(self):
        return f"RadialFunc(in_dim={self.in_dim}, out_dim={self.out_dim})"

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, self.out_dim, 1, self.in_dim, 1, self.num_freq)


class PairwiseConv(nn.Module):
    """SO(3)-equivariant convolution between two single-type features"""
    def __init__(self, degree_in: int, nc_in: int, degree_out: int,
                 nc_out: int, input_feature_size: int):
        """SO(3)-equivariant convolution between a pair of feature types.

        This layer performs a convolution from nc_in features of type degree_in
        to nc_out features of type degree_out.

        Args:
            degree_in: degree of input fiber
            nc_in: number of channels on input
            degree_out: degree of out order
            nc_out: number of channels on output
        """
        super().__init__()
        # Log settings
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out
        self.input_feature_size = input_feature_size

        # Functions of the degree
        self.num_freq = 2*min(degree_in, degree_out) + 1
        self.d_out = 2*degree_out + 1

        # Radial profile function
        self.rp = RadialFunc(self.num_freq, nc_in, nc_out)

    @profile
    def forward(self, feat, basis):
        # Get radial weights
        R = self.rp(feat)
        kernel = torch.sum(R * basis[f'{self.degree_in},{self.degree_out}'], -1)
        return kernel.view(feat.shape[0], self.input_feature_size, -1) #


class S1x1SO3(nn.Module):
    """Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.

    This is equivalent to a self-interaction layer in TensorField Networks.
    """
    def __init__(self, batch_size, f_in, f_out, learnable=True):
        """SO(3)-equivariant 1x1 convolution.

        Args:
            f_in: input Fiber() of feature multiplicities and types
            f_out: output Fiber() of feature multiplicities and types
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.batch_size =batch_size

        # Linear mappings: 1 per output feature type
        self.transform = nn.ParameterDict()
        #for m_out, d_out in self.f_out.structure:
        m_in = self.f_in.structure_dict[0]
        m_out = self.f_out.structure_dict[0] * 4 * 2
        self.transform = nn.Parameter(torch.randn(self.batch_size, m_in, m_out) / np.sqrt(m_in), requires_grad=learnable)

    def __repr__(self):
         return f"S1x1SO3(structure={self.f_out})"

    def forward(self, features, **kwargs):
        #output = {}
        #for k, v in features.items():
            #if str(k) in self.transform.keys():
        #print(self.transform.shape, features.shape)
        output = torch.matmul(features, self.transform)
        return output


class SNormBias(nn.Module):
    """Norm-based SO(3)-equivariant nonlinearity with only learned biases."""

    def __init__(self, fiber, nonlin=nn.ReLU(inplace=True),
                 num_layers: int = 0):
        """Initializer.

        Args:
            fiber: Fiber() of feature multiplicities and types
            nonlin: nonlinearity to use everywhere
            num_layers: non-negative number of linear layers in fnc
        """
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.num_layers = num_layers

        # Regularization for computing phase: gradients explode otherwise
        self.eps = 1e-12

        # Norm mappings: 1 per feature type
        self.bias = nn.ParameterDict()
        for m, d in self.fiber.structure:
            self.bias[str(d)] = nn.Parameter(torch.randn(m).view(1, m))

    def __repr__(self):
        return f"SNormTFN()"


    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # Compute the norms and normalized features
            # v shape: [...,m , 2*k+1]
            norm = v.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(v)
            phase = v / norm

            # Transform on norms
            # transformed = self.transform[str(k)](norm[..., 0]).unsqueeze(-1)
            transformed = self.nonlin(norm[..., 0] + self.bias[str(k)])

            # Nonlinearity on norm
            output[k] = (transformed.unsqueeze(-1) * phase).view(*v.shape)

        return output


class SNormSO3(nn.Module):
    """spherical Norm-based SO(3)-equivariant nonlinearity.

    Nonlinearities are important in SO(3) equivariant SCNs. They are also quite
    expensive to compute, so it is convenient for them to share resources with
    other layers, such as normalization. The general workflow is as follows:

    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase

    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.
    """
    def __init__(self, fiber, nonlin=nn.ReLU(inplace=True)):
        """Initializer.

        Args:
            fiber: Fiber() of feature multiplicities and types
            nonlin: nonlinearity to use everywhere
            num_layers: non-negative number of linear layers in fnc
        """
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin

        # Regularization for computing phase: gradients explode otherwise
        self.eps = 1e-12

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for m, d in self.fiber.structure:
            self.transform = self._build_net(m)

    def __repr__(self):
         return f"SNormSO3(nonlin={self.nonlin})"

    def _build_net(self, m: int):
        net = []
        net.append(BN(int(m)))
        net.append(self.nonlin)
        return nn.Sequential(*net)

    @profile
    def forward(self, features, **kwargs):
        # Compute the norms and normalized features
        norm = features.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(features)
        phase = features / norm

        # Transform on norms
        transformed = self.transform(norm)

        # Nonlinearity on norm
        output = (transformed * phase).view(*features.shape)

        return output


class BN(nn.Module):
    """SE(3)-equvariant batch/layer normalization"""
    def __init__(self, m):
        """SE(3)-equvariant batch/layer normalization

        Args:
            m: int for number of output channels
        """
        super().__init__()
        self.bn = nn.LayerNorm(m)

    def forward(self, x):
        return self.bn(x)


class SConvSO3Partial(nn.Module):
    """Spherical SO(3)-equivariant layer"""
    def __init__(self, f_in, f_out, input_feature_size: int=0, x_ij=None):
        """SO(3)-equivariant partial convolution.
        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
        """
        super().__init__()
        self.f_out = f_out
        self.input_feature_size = input_feature_size

        # adding/concatinating relative position to feature vectors
        # 'cat' concatenates relative position & existing feature vector
        # 'add' adds it, but only if multiplicity > 1
        assert x_ij in [None, 'cat', 'add']
        self.x_ij = x_ij
        if x_ij == 'cat':
            self.f_in = Fiber.combine(f_in, Fiber(structure=[(1,1)]))
        else:
            self.f_in = f_in

        # compute W_lk
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, input_feature_size)

    def __repr__(self):
        return f'SConvSO3Partial(structure={self.f_out})'

    @profile
    def forward(self, h, S=None, r=None, basis=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            h: dict of node-features
            G: minibatch of (homo)graphs
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, features_in, features_out]
        """
        feat = torch.cat([r, ], -1)
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                etype = f'({di},{do})'
                S[f'out{do}'] = torch.matmul(h['d'],self.kernel_unary[etype](feat, basis))
        return {f'{d}': S[f'out{d}'] for d in self.f_out.degrees}


class SMABSO3(nn.Module):
    """A SO(3)-equivariant multi-headed self-attention module."""
    def __init__(self, f_value: Fiber):
        """SO(3)-equivariant MAB (multi-headed attention block) layer.

        Args:
            f_value: Fiber() object for value-embeddings
        """
        super().__init__()
        self.f_value = f_value

    def __repr__(self):
        return f'SMABSO3(structure={self.f_value})'

    @profile
    def forward(self, v, k: Dict=None, q: Dict=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            v: dict of value features
            k: dict of key features
            q: dict of query features
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        #output = {}
        e = {}
        alpha = {}
        for m, d in self.f_value.structure:
        # Compute attention weights
        ## Inner product between (key) neighborhood and (query) center
            k[f'{d}'] = torch.Tensor.permute(k[f'{d}'], (0, 2, 1))
            e[f'{d}'] = torch.matmul(q[f'{d}'], k[f'{d}'])/np.sqrt(q[f'{d}'].shape[2])
        # Apply softmax
            alpha[f'{d}'] = torch.nn.functional.softmax(e[f'{d}'], dim=1)
        for m, d in self.f_value.structure:
            v[f'out{d}'] = torch.matmul(alpha[f'{d}'], v[f'{d}'])
            #output[f'{d}'] = v[f'out{d}']
            if d == 0:
             output = v[f'out{d}']
            else:
             output = torch.cat((output,v[f'out{d}']),2)

        return output


class SSO3Res(nn.Module):
    """Spherical attention block with SO(3)-equivariance and skip connection"""
    def __init__(self, f_in: Fiber, f_out: Fiber, input_feature_size: int=0, div: float=4, batch_size: int=8, nclass: int=10, learnable_skip=True, x_ij=None):
        super().__init__()
        self.batch_size = batch_size
        self.nclass = nclass
        self.f_in = f_in
        self.f_out = f_out
        self.div = div
        #self.skip = skip  # valid: 'cat', 'sum', None
        # f_mid_out has same structure as 'f_out' but #channels divided by 'div'
        # this will be used for the values
        f_mid_out = {k: int(v // div) for k, v in self.f_out.structure_dict.items()}
        self.f_mid_out = Fiber(dictionary=f_mid_out)

        self.input_feature_size = input_feature_size

        self.SMAB = nn.ModuleDict()

        # Projections
        self.SMAB['v'] = SConvSO3Partial(f_in, self.f_mid_out, input_feature_size=input_feature_size, x_ij=x_ij)
        self.SMAB['k'] = SConvSO3Partial(f_in, self.f_mid_out, input_feature_size=input_feature_size, x_ij=x_ij)
        self.SMAB['q'] = SConvSO3Partial(f_in, self.f_mid_out, input_feature_size=input_feature_size, x_ij=x_ij)

        # Attention
        self.SMAB['attn'] = SMABSO3(self.f_mid_out)

        # Skip connections
        self.project = S1x1SO3(self.batch_size, f_in, f_out, learnable=learnable_skip)
        #self.project = nn.Sequential(nn.Linear(self.input_feature_size, f_out.structure_dict[0] * 4 * 2),
                                     #nn.LayerNorm(f_out.structure_dict[0] * 4 * 2),
                                     #nn.ReLU(inplace=True),
                                     #nn.Linear(f_out.structure_dict[0] * 4 * 2, f_out.structure_dict[0] * 4 * 2),
                                     #)
        #project
        total = nclass*16

        for i in range(3):
            total+= nclass*16*(2*(i+1)+1)

        self.linear = nn.Sequential(nn.Linear(total, f_out.structure_dict[0] * 4 * 2),
                                    nn.LayerNorm(f_out.structure_dict[0] * 4 * 2),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(f_out.structure_dict[0] * 4 * 2, f_out.structure_dict[0] * 4 * 2),
                      )
        nn.init.kaiming_uniform_(self.linear[0].weight)
        nn.init.kaiming_uniform_(self.linear[3].weight)
        #nn.init.kaiming_uniform_(self.project[0].weight)
        #nn.init.kaiming_uniform_(self.project[3].weight)
    @profile
    def forward(self, features, S, **kwargs):
        # Embeddings
        v = self.SMAB['v'](features, S=S, **kwargs)
        k = self.SMAB['k'](features, S=S, **kwargs)
        q = self.SMAB['q'](features, S=S, **kwargs)

        # Attention
        z = self.SMAB['attn'](v, k=k, q=q)

        # Skip + residual
        features['d'] = self.project(features['d'])

        # project
        z = self.linear(z) # 2560 (160+480+800+1120) -> 512

        z = z + features['d']
        return z









