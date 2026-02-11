import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10) -> str:
    H = np.asarray(hessian, dtype=float)
    
    if H.ndim !=2 or H.shape[0] != H.shape[1]:
        raise ValueError("Hessian matrix must be square (n x n).")

    eigenvalues = np.linalg.eigvalsh(H)

    if np.any(np.abs(eigenvalues) < tol):
        return "inconclusive"
    
    if np.all(eigenvalues > 0):
        return "local_minimum"

    if np.all(eigenvalues < 0):
        return "local_maximum"
        
    return "saddle_point"