import numpy as np

def iterateQR(Q, R, A):

    maxIterations = 100
    eigenvectors = Q

    for i in range(0, maxIterations):
        T = R * Q
        (Q, R) = gramSchmidtQR(T)
        eigenvectors = eigenvectors * Q

    eigenvalues = np.zeros((Q.shape[0]))

    for i in range(0, A.shape[0]):
        eigenvalues[i] = T[i, i]
    return eigenvalues, eigenvectors.A

def gramSchmidtQR(A):
    m = A.shape[0]
    n = A.shape[1]
    Q = np.matrix(np.zeros((m, m)))
    R = np.matrix(np.zeros((m, n)))
    A1 = np.matrix(A)
    for k in xrange(n):
        V = A1[0:m, k]
        R[k,k] = np.linalg.norm(V)
        Q[:, k] = V/R[k,k]
        for j in xrange(k+1, n):
            R[k,j] = Q[:,k].T * A1[0:m, j]
            A1[0:m,j] = A1[0:m,j] - Q[:, k] * R[k,j]
    return (Q,R)

def eigen(A):
    if A.shape[0] == A.shape[1]:
        (Q, R) = gramSchmidtQR(A)
        return iterateQR(Q,R,A)
    else:
        print("The matrix must be square")
