import torch
def orthonorm_op(x, epsilon=1e-7):
    '''
    Computes a matrix that orthogonalizes the input matrix x

    x:      an n x d input matrix
    eps:    epsilon to prevent nonzero values in the diagonal entries of x

    returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                right multiplication
    '''
    x_2 = torch.mm(x.t(), x)
    x_2 += torch.eye(x.shape[1])*epsilon
    L = torch.cholesky(x_2)
    # ortho_weights = tf.transpose(tf.matrix_inverse(L)) * tf.sqrt(tf.cast(tf.shape(x)[0], dtype=K.floatx()))
    return ortho_weights

def Orthonorm(x, name=None):
    '''
    Builds keras layer that handles orthogonalization of x

    x:      an n x d input matrix
    name:   name of the keras layer

    returns:    a keras layer instance. during evaluation, the instance returns an n x d orthogonal matrix
                if x is full rank and not singular
    '''
    # get dimensionality of x
    d = x.get_shape().as_list()[-1]
    # compute orthogonalizing matrix
    ortho_weights = orthonorm_op(x)
    # create variable that holds this matrix
    ortho_weights_store = K.variable(np.zeros((d,d)))
    # create op that saves matrix into variable
    ortho_weights_update = tf.assign(ortho_weights_store, ortho_weights, name='ortho_weights_update')
    # switch between stored and calculated weights based on training or validation
    l = Lambda(lambda x: K.in_train_phase(K.dot(x, ortho_weights), K.dot(x, ortho_weights_store)), name=name)

    l.add_update(ortho_weights_update)
    return l


if __name__ == '__main__':
    x = torch.randn(3, 3)
    x_2 = torch.mm(x.t(), x)
    # x_2 += torch.eye(x.shape[1]) * (1e-7)
    L = torch.cholesky(x_2)
    R = torch.mm(x ,L.inverse().t())
    # print(x)
    # print(x_2)
    # print(L)
    # print(R)
    print(torch.mm(R.t(), R))