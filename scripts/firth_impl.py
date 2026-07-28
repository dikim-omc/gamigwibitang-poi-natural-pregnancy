import numpy as np

def firth_logistic(X, y, weights=None, max_iter=100, tol=1e-6):
    n, p = X.shape
    X = np.column_stack([np.ones(n), X])
    p_full = p + 1
    beta = np.zeros(p_full)
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=float)

    for iteration in range(max_iter):
        eta = X @ beta
        pi = 1 / (1 + np.exp(-eta))
        W = weights * pi * (1 - pi)
        W = np.clip(W, 1e-10, None)

        WX = X * W[:, None]
        I = X.T @ WX
        I_inv = np.linalg.inv(I)

        sqrtW = np.sqrt(W)
        XWsqrt = X * sqrtW[:, None]
        H = XWsqrt @ I_inv @ XWsqrt.T
        h = np.diag(H)

        U = X.T @ (weights * (y - pi) + h * (0.5 - pi))
        delta = I_inv @ U
        beta_new = beta + delta
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta
    pi = 1 / (1 + np.exp(-eta))
    W = np.clip(weights * pi * (1 - pi), 1e-10, None)
    I = X.T @ (X * W[:, None])
    cov = np.linalg.inv(I)
    return beta, cov, iteration + 1
