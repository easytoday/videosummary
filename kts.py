import numpy as np

def cpd_nonlin(K, ncp, lmin=1, lmax=10000, backtrack=True):
    m = K.shape[0]
    V = np.zeros((ncp + 1, m + 1))
    V.fill(np.inf)
    V[0, 0] = 0
    
    # Matrice de somme cumulative pour calculer les coûts rapidement
    I = np.zeros((m + 1, m + 1))
    for i in range(m):
        for j in range(i, m):
            # Coût basé sur la variance intra-segment
            I[i+1, j+1] = K[i, i] + K[j, j] - 2 * K[i, j]

    pos = np.zeros((ncp + 1, m + 1), dtype=int)

    for k in range(1, ncp + 1):
        for j in range(k * lmin, m + 1):
            for i in range((k - 1) * lmin, j - lmin + 1):
                cost = V[k - 1, i] + I[i + 1, j]
                if cost < V[k, j]:
                    V[k, j] = cost
                    pos[k, j] = i

    if not backtrack:
        return V, pos

    cps = np.zeros(ncp, dtype=int)
    curr = m
    for k in range(ncp, 0, -1):
        cps[k - 1] = pos[k, curr]
        curr = cps[k - 1]
    
    return cps
