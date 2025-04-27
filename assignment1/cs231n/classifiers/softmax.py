from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W) # scores shape = (C, )

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)
        loss -= logp[y[i]]  # negative log probability is the loss
        
        # gradient 
        for j in range(num_classes):
          dW[:, j] += X[i] * p[j] # dW[:, j] shape = (D, 1) 
        
        dW[:, y[i]] -= X[i]


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    scores = np.dot(X, W) # scores shape = (N, C)
    scores -= np.max(scores, axis = 1, keepdims = True)
    p = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)
    logp = np.log(p)

    loss -= np.sum(logp[np.arange(X.shape[0]), y])
    loss = loss / X.shape[0] + reg * np.sum(W * W)

    p[np.arange(X.shape[0]), y] -= 1
    dW = X.T.dot(p)
    dW = dW / X.shape[0] + reg * W

    return loss, dW
