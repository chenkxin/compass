import torch
import torch.nn as nn
class Cholesky(nn.Module):
    def __init__(self):
        super(Cholesky, self).__init__()

    def forward(self, x):
        return self.orthonorm(x)

    def orthonorm(self, x, epsilon=1e-7):
        '''
        Computes a matrix that orthogonalizes the input matrix x

        x:      an n x d input matrix
        eps:    epsilon to prevent nonzero values in the diagonal entries of x

        returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                    right multiplication
        '''
        x_2 = torch.mm(x.t(), x)
        x_2 += torch.eye(x.shape[1]).cuda() * torch.tensor(epsilon, device='cuda')
        L = torch.cholesky(x_2)
        R = torch.mm(x, L.inverse().t())
        return R

def cholesky(x, epsilon=1e-7):
    '''
    Computes a matrix that orthogonalizes the input matrix x

    x:      an n x d input matrix
    eps:    epsilon to prevent nonzero values in the diagonal entries of x

    returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                right multiplication
    '''
    x_2 = torch.mm(x.t(), x)
    x_2 += torch.eye(x.shape[1]).cuda() * torch.tensor(epsilon, device='cuda')
    L = torch.cholesky(x_2)
    R = torch.mm(x, L.inverse().t())
    return R

if __name__ == '__main__':
    x = torch.randn(3, 3)
    cholesky = Cholesky()
    R = cholesky.forward(x)
    print(torch.mm(R.t(), R))