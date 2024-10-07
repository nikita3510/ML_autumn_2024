import numpy as np
import scipy
from scipy.special import expit


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax = self.matvec_Ax(x)
        log_likelihood = np.logaddexp(0, -self.b * Ax)
        return np.mean(log_likelihood) + (self.regcoef / 2) * np.linalg.norm(x)**2

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        p = expit(-self.b * Ax)
        grad_log_likelihood = -self.matvec_ATx(self.b * p) / len(self.b)
        return grad_log_likelihood + self.regcoef * x

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        p = expit(-self.b * Ax)
        diag_p = p * (1 - p)
        hessian_log_likelihood = self.matmat_ATsA(diag_p) / len(self.b)
        return hessian_log_likelihood + self.regcoef * np.eye(len(x))

def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):

    if isinstance(A, csr_matrix):
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        matmat_ATsA = lambda s: A.T.dot(csr_matrix(np.diag(s))).dot(A)
    else:
        matvec_Ax = lambda x: A @ x
        matvec_ATx = lambda x: A.T @ x
        matmat_ATsA = lambda s: A.T @ np.diag(s) @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)

    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

def grad_finite_diff(func, x, eps=1e-8):

    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        grad[i] = (func(x + eps * e_i) - func(x)) / eps
    return grad

def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            hess[i, j] = (func(x + eps * e_i + eps * e_j) -
                          func(x + eps * e_i) -
                          func(x + eps * e_j) +
                          func(x)) / (eps ** 2)
            hess[j, i] = hess[i, j]
    return hess