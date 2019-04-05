
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.misc import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
y = boston['target']

N = x.shape[0]
d = x.shape[1]
idx = np.random.permutation(range(N))


def l2(A, B):
    '''
    Compute L2 norm between each row of matrices A and B
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Implements Locally Reweighted Least Squares
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    N = x_train.shape[0]
    d = x_train.shape[1]

    a = np.zeros((N,1)) #initialize weights
    dist = l2(test_datum.transpose(), x_train) #matrix of norms for weight calculation
    for i in range(N):  #compute weights
        a[i] = np.exp(logsumexp(-dist[0][i]/(2*tau**2))) / np.exp(logsumexp(-dist/(2*tau**2)))

    w = np.linalg.solve((x_train.transpose()) @ (a*np.identity(N)) @ (x_train) + (lam*np.identity(d)), (x_train.transpose()) @ (a*np.identity(N)) @ (y_train))
    y_hat = test_datum.transpose() @ w
    return y_hat


def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    d = x.shape[1]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = val_frac)
    y_train = y_train.reshape(y_train.size, 1)
    y_valid = y_valid.reshape(y_valid.size, 1)

    loss_train = np.zeros(taus.size)
    loss_valid = np.zeros(taus.size)
    for i in range(taus.size):
        for j in range(x_train.shape[0]):
            loss_train[i] += (1/2) * (LRLS(x_train[j].reshape(d,1), x_train, y_train, taus[i]) - y_train[j])**2
        for j in range(x_valid.shape[0]):
            loss_valid[i] += (1/2) * (LRLS(x_valid[j].reshape(d,1), x_train, y_train, taus[i]) - y_valid[j])**2
    loss_train /= x_train.shape[0]
    loss_valid /= x_valid.shape[0]
    return loss_train, loss_valid


if __name__ == "__main__":
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(train_losses, label='train loss')
    plt.semilogx(test_losses, label = 'test loss')
    plt.legend()
    plt.show()
    plt.savefig('./Desktop/train_test_loss.png')

