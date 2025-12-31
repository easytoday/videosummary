import numpy as np
from knapsack import knapsack_dp
import math

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Génère un résumé binaire basé sur des segments (keyshots)."""
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    
    # S'assurer que les positions couvrent toute la vidéo
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    
    # Mapper les scores prédits sur toutes les frames de la vidéo
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i >= len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    # Calculer le score moyen pour chaque segment (shot)
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    # Limite de 15% de la durée totale
    limits = int(math.floor(n_frames * proportion))

    if method == 'knapsack':
        # On choisit les segments qui maximisent le score sous la contrainte de durée
        picks = knapsack_dp(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError(f"Méthode inconnue {method}")

    # Construire le vecteur binaire final
    summary = []
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            summary.append(np.ones(nf, dtype=np.float32))
        else:
            summary.append(np.zeros(nf, dtype=np.float32))
    
    summary = np.concatenate(summary)
    return summary

def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Calcule le F-score en comparant le résumé machine aux résumés humains."""
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users, n_frames = user_summary.shape

    # Binarisation
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    # Ajustement de la longueur si nécessaire
    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        machine_summary = np.pad(machine_summary, (0, n_frames - len(machine_summary)), 'constant')

    f_scores, prec_arr, rec_arr = [], [], []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
            
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':
        return np.mean(f_scores), np.mean(prec_arr), np.mean(rec_arr)
    else:
        max_idx = np.argmax(f_scores)
        return f_scores[max_idx], prec_arr[max_idx], rec_arr[max_idx]
