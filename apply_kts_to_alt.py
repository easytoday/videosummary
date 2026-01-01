import h5py
import numpy as np

# --- ALGORITHME KTS (Version simplifiée pour la vidéo) ---
def vsum_kts(K, ncp):
    """
    K : Matrice de noyau (n_feats x n_feats)
    ncp : Nombre de points de coupure souhaités
    """
    n = K.shape[0]
    # Calcul des sommes cumulées pour accélérer le calcul de la variance
    D = np.cumsum(np.diag(K))
    S = np.cumsum(np.cumsum(K, axis=0), axis=1)
    
    def get_cost(i, j):
        # Calcule la variance intra-segment entre l'indice i et j
        # Plus la variance est faible, plus le segment est homogène
        sum_k = S[j, j]
        if i > 0: sum_k -= (S[i-1, j] + S[j, i-1] - S[i-1, i-1])
        diag_k = D[j]
        if i > 0: diag_k -= D[i-1]
        return diag_k - sum_k / (j - i + 1)

    # Programmation dynamique pour trouver les coupures optimales
    V = np.full((ncp + 1, n), np.inf)
    P = np.zeros((ncp + 1, n), dtype=int)
    
    for j in range(n):
        V[0, j] = get_cost(0, j)
        
    for k in range(1, ncp + 1):
        for j in range(k, n):
            for i in range(k-1, j):
                cost = V[k-1, i] + get_cost(i+1, j)
                if cost < V[k, j]:
                    V[k, j] = cost
                    P[k, j] = i
                    
    # Reconstruction des points de coupure
    cps = []
    curr = n - 1
    for k in range(ncp, 0, -1):
        curr = P[k, curr]
        cps.append(curr)
    return sorted(cps)

# --- SCRIPT DE RESTRUCTURATION ---
INPUT_PATH = "dataset_cpu/features.h5"
OUTPUT_PATH = "dataset_cpu/alt_dataset_kts.h5"

with h5py.File(INPUT_PATH, 'r') as f_in, h5py.File(OUTPUT_PATH, 'w') as f_out:
    for key in f_in.keys():
        print(f"➜ Segmentation KTS : Vidéo {key}")
        # Récupération des features
        features = f_in[key][...]
        if len(features.shape) == 1: # Sécurité si le dataset est mal aplati
            continue
            
        n_feats = features.shape[0]
        
        # 1. Calcul de la matrice de similarité (Kernel)
        K = np.dot(features, features.T)
        
        # 2. Détermination du nombre de segments (Heuristique de Zhou : ~1 segment toutes les 25 features)
        ncp = max(1, int(n_feats / 25))
        
        # 3. Calcul des points de coupure
        cps = vsum_kts(K, ncp)
        
        # 4. Création des change_points à l'échelle des frames (x15)
        full_cps = [0] + [c + 1 for c in cps] + [n_feats]
        change_points = []
        for i in range(len(full_cps) - 1):
            start = full_cps[i] * 15
            end = (full_cps[i+1] * 15) - 1
            change_points.append([start, end])
            
        # 5. Sauvegarde dans la structure officielle de Zhou
        g = f_out.create_group(key)
        g.create_dataset('features', data=features.astype(np.float32))
        g.create_dataset('change_points', data=np.array(change_points, dtype=np.int32))
        g.create_dataset('n_frame_per_seg', data=np.array([cp[1]-cp[0]+1 for cp in change_points], dtype=np.int32))
        g.create_dataset('n_frames', data=np.int32(n_feats * 15))
        g.create_dataset('picks', data=np.arange(0, n_feats * 15, 15, dtype=np.int32))
        
        # Pour l'évaluation (même si vide pour l'instant)
        g.create_dataset('gtscore', data=np.zeros(n_feats * 15, dtype=np.float32))
        g.create_dataset('user_summary', data=np.zeros((1, n_feats * 15), dtype=np.float32))

print(f"\n✅ Succès ! Votre dataset '{OUTPUT_PATH}' est prêt et segmenté via KTS.")
