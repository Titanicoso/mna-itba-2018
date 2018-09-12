import numpy as np

def iterateQR(A):

    maxIterations = 100
    eigenvectors = np.identity(A.shape[0])
    T = A

    for i in range(maxIterations):
        Q, R = householderQR(T)
        T = R.dot(Q)
        eigenvectors = eigenvectors.dot(Q)
        if np.allclose(T, np.triu(T), atol = 1e-4):
            break

    eigenvalues = np.diag(T)

    sort = np.argsort(np.absolute(eigenvalues))[::-1]
    return eigenvalues[sort], eigenvectors[sort]

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

def householderQR(A):
    m,n = A.shape
    R = np.matrix(A)
    Q = np.identity(m)

    for k in range(n):
        norm = np.linalg.norm(R[k:,k])
        s = -np.sign(R[k,k])
        u = R[k,k] - s * norm
        w = R[k:,k]/u
        w[0] = 1
        tau = -s * u/norm
        R[k:, :] = R[k:, :] - (tau*w) * w.T.dot(R[k:, :])
        Q[:, k:] = Q[:, k:] - (Q[:,k:] * w).dot((tau * w).T)
    return Q, R

def eigen(A):
    if A.shape[0] == A.shape[1]:
        return iterateQR(A)
    else:
        print("The matrix must be square")

def rsvAndEigenValues(A):
    m,n = A.shape
    if n > m:
        aux = A.dot(A.T)
        S, U = eigen(aux)
        S = np.sqrt(S)
        V = A.T.dot(U)
        S1 = np.diag(S)
        for k in range(S1.shape[0]):
            S1[k,k] = 1/S1[k,k]

        V = V.dot(S1)
        return S, np.asmatrix(V.T)
    aux = A.T.dot(A)
    S,V = eigen(aux)
    S = np.sqrt(S)
    return S, np.asmatrix(V)