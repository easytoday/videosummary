import torch
import sys

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False, reward_type='DR-DSN'):
    """
    Calcule la récompense basée sur la diversité (diversity) et la représentativité (representativeness)
    comme décrit dans la section "Diversity-Representativeness Reward Function"
    
    D'après l'article, un résumé de haute qualité devrait être à la fois diversifié et représentatif.
    La fonction de récompense DR évalue la qualité du résumé en combinant deux récompenses :
    1. R_div : mesure le degré de diversité entre les images sélectionnées (à quel point elles sont différentes)
    2. R_rep : mesure le degré de représentativité des images sélectionnées pour l'ensemble de la vidéo
    
    Cette fonction calcule :
    R(S) = R_div + R_rep  (Formule (6) de l'article)
    
    Ceci est au cœur du cadre d'apprentissage par renforcement décrit dans la Figure 1,
    où la récompense R(S) est calculée en fonction de la qualité du résumé et est utilisée
    pour entraîner le DSN via le gradient de politique.
    
    Paramètres :
        seq : séquence de caractéristiques, taille (1, seq_len, dim)
        actions : séquence d'actions binaires, taille (1, seq_len, 1)
        ignore_far_sim (bool) : ignorer ou non la similarité temporelle lointaine (défaut : True)
                              définir d(x_i, x_j) = 1 si |i - j| > λ comme mentionné dans l'introduction de l'article
        temp_dist_thre (int) : seuil λ pour ignorer la similarité temporelle lointaine (défaut : 20)
                              Comme décrit dans la section "Implementation details" : λ est fixé à 20
        use_gpu (bool) : utiliser ou non le GPU
        reward_type (str) : type de récompense à utiliser ('dr', 'd', 'r', 'd-nolambda')
                          'dr' : utilise à la fois la diversité et la représentativité (DR-DSN)
                          'd' : utilise uniquement la diversité (D-DSN)
                          'r' : utilise uniquement la représentativité (R-DSN)
                          'd-nolambda' : utilise uniquement la diversité mais sans ignorer la similarité temporelle lointaine
                          
                          Ces variantes sont évaluées dans les Tableaux 1 et 2 de l'article
    
    Retourne :
        reward : récompense agrégée selon le reward_type
    """
    # Détacher les gradients pour le calcul de la récompense
    _seq = seq.detach()
    _actions = actions.detach()
    
    # Définir le device en fonction de la séquence d'entrée pour garantir la cohérence
    device = seq.device
    
    # Obtenir les indices des images sélectionnées (valeurs 1 dans actions)
    # Correspond à S = {i|a_i = 1, i = 1,2,...} dans l'article
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    if num_picks == 0:
        # Retourner une récompense de 0 si aucune image n'est sélectionnée
        # Comme mentionné dans l'article : "We give zero reward to DSN when no frames are selected"
        # Ceci fait partie de l'entraînement du DSN - il ne devrait pas y avoir de cas où aucune image n'est sélectionnée
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.to(device)
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # Calculer la récompense de diversité (diversity reward - R_div)
    if num_picks == 1:
        # Si une seule image est sélectionnée, il n'y a pas de diversité car il n'y a pas deux images à comparer
        # Selon la formule (3), on ne peut pas calculer le degré de diversité avec une seule image
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.to(device)
    else:
        # Normaliser les vecteurs de caractéristiques pour calculer la similarité cosinus
        # Cela prépare le calcul de d(x_i, x_j) = 1 - cos(x_i, x_j)
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        
        # Calculer la matrice de dissimilarité (Formule (4) dans l'article)
        # d(x_i, x_j) = 1 - (x_i·x_j)/(||x_i||·||x_j||)
        # Où :
        # - d(x_i, x_j) est la différence entre les images i et j
        # - (x_i·x_j)/(||x_i||·||x_j||) est la similarité cosinus
        # Visuellement : plus la valeur est proche de 1, plus les images sont différentes
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        
        # Extraire la sous-matrice contenant uniquement les images sélectionnées pour le résumé
        # C'est la matrice d(x_i, x_j) avec i,j ∈ S (les images sélectionnées)
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        
        if ignore_far_sim:
            # Dans l'article, il est mentionné dans l'introduction : définir d(x_i, x_j) = 1 si |i - j| > λ
            # C'est une hypothèse importante : les images éloignées dans le temps
            # doivent être considérées comme complètement différentes pour garantir la diversité du résumé
            # et éviter de se concentrer trop sur une partie de la vidéo
            pick_mat = pick_idxs.unsqueeze(0).expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            
            # Définir la valeur à 1 (différence complète) pour les paires d'images éloignées
            # λ = temp_dist_thre (par défaut 20 comme dans la section "Implementation details")
            # C'est la mise en œuvre de la règle : d(x_i, x_j) = 1 si |i - j| > λ
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        
        # Calculer la récompense de diversité (Formule (3) dans l'article)
        # R_div = (∑_{i∈S} ∑_{j∈S,j≠i} d(x_i,x_j))/(|S|·(|S|-1))
        # Où :
        # - S = {i|a_i = 1, i = 1,2,...} est l'ensemble des images sélectionnées
        # - |S| est le nombre d'images sélectionnées (num_picks)
        # - d(x_i,x_j) est la différence entre deux images i et j
        # 
        # Visuellement : R_div est d'autant plus élevé que les images sélectionnées sont différentes
        # Objectif : Encourager le résumé à contenir des images diverses, sans répétition
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

    # Calculer la récompense de représentativité (representativeness reward - R_rep)
    
    # Calculer la matrice des distances euclidiennes au carré entre les vecteurs de caractéristiques
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    # dist_mat = taille (n, n) avec n le nombre d'images
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)  # ||x_i||^2
    dist_mat = dist_mat + dist_mat.t()  # ||x_i||^2 + ||x_j||^2
    dist_mat.addmm_(beta=1, mat1=_seq, mat2=_seq.t(), alpha=-2)  # soustraire 2*x_i·x_j
    
    # Obtenir la distance de chaque image de la vidéo aux images sélectionnées pour le résumé
    # dist_mat[:,pick_idxs] : distance de chaque image aux images sélectionnées
    # Taille (n, num_picks) ou (n,) si num_picks = 1
    # C'est la première étape pour calculer min_{j∈S} ||x_i - x_j||^2 dans la formule (5)
    dist_mat = dist_mat[:,pick_idxs]
    
    # Trouver la distance minimale de chaque image aux images sélectionnées
    # min_{j∈S} ||x_i - x_j||^2 avec S l'ensemble des images sélectionnées
    # Cela mesure le degré de "proximité" auquel chaque image de la vidéo
    # peut être représentée par au moins une image du résumé
    if num_picks > 1:
        # Trouver la distance minimale le long de la dimension 1 (la dimension des images sélectionnées)
        dist_mat = dist_mat.min(1, keepdim=True)[0]
    else:
        # S'il n'y a qu'une seule image sélectionnée, pas besoin de trouver le min
        dist_mat = dist_mat.view(-1, 1)
    
    # Calculer la récompense de représentativité (Formule (5) dans l'article)
    # R_rep = exp(-1/T·∑_{i=1}^T min_{j∈S}||x_i - x_j||^2)
    # Où :
    # - T est le nombre total d'images
    # - x_i est la caractéristique de l'image i
    # - S = {i|a_i = 1, i = 1,2,...} est l'ensemble des images sélectionnées
    # - min_{j∈S}||x_i - x_j||^2 est la distance minimale de l'image i à n'importe quelle image sélectionnée
    #
    # Visuellement : R_rep est d'autant plus élevé que les images sélectionnées sont proches de toutes les autres images
    # Objectif : Encourager la sélection d'images représentatives pour l'ensemble du contenu vidéo
    #
    # Explication détaillée :
    # 1. Calculer la distance minimale moyenne : 1/T·∑_{i=1}^T min_{j∈S}||x_i - x_j||^2
    #    C'est le degré moyen auquel chaque image de la vidéo est représentée par le résumé
    # 2. Prendre exp(-) pour transformer la distance en similarité (plus la distance est grande, plus la valeur est petite)
    #    Si la distance moyenne est petite (le résumé est bien représentatif) alors R_rep sera élevé
    reward_rep = torch.exp(-dist_mat.mean())
    
    # S'assurer que reward_div et reward_rep sont sur le même device que l'entrée
    reward_div = reward_div.to(device)
    reward_rep = reward_rep.to(device)
    
    # Combiner les deux récompenses selon le type de récompense demandé
    # Ces types de récompenses sont évalués dans le Tableau 1 de l'article, comparant l'efficacité de chaque type
    if reward_type == 'dr':  # Utiliser les deux récompenses (DR-DSN)
        # R(S) = R_div + R_rep (Formule (6) dans l'article)
        # Dans l'article, les auteurs soulignent que : "R_div et R_rep se complètent et travaillent ensemble
        # pour guider le DSN" et "DR-DSN surpasse D-DSN et R-DSN sur les deux jeux de données"
        # 
        # Multiplier par 0.5 pour équilibrer les deux récompenses, garder le niveau global similaire
        reward = (reward_div + reward_rep) * 0.5
    elif reward_type == 'd':  # Utiliser uniquement la récompense de diversité (D-DSN)
        # Utiliser uniquement R_div pour évaluer l'efficacité de la seule récompense de diversité
        # Dans le Tableau 1, D-DSN donne de meilleurs résultats que R-DSN, mais pas aussi bon que DR-DSN
        reward = reward_div
    elif reward_type == 'r':  # Utiliser uniquement la récompense de représentativité (R-DSN)
        # Utiliser uniquement R_rep pour évaluer l'efficacité de la seule récompense de représentativité
        reward = reward_rep
    elif reward_type == 'd-nolambda':  # D-DSN sans lambda (sans ignorer la similarité temporelle lointaine)
        # Pour ce type de récompense, nous devons recalculer reward_div avec ignore_far_sim=False
        # Le but est de vérifier l'efficacité de la règle d(x_i, x_j) = 1 si |i - j| > λ
        # S'assurer d'utiliser toutes les images, sans ignorer les images éloignées dans le temps
        
        # Si précédemment reward_div a été calculé avec ignore_far_sim=True, nous devons le recalculer
        if ignore_far_sim and num_picks > 1:
            # Recalculer la matrice dissim_submat sans appliquer la technique lambda
            normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
            dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
            dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
            
            # NE PAS utiliser la technique lambda ici (ignore_far_sim = False)
            # Recalculer reward_div sans lambda
            reward_div_nolambda = dissim_submat.sum() / (num_picks * (num_picks - 1.))
            reward = reward_div_nolambda
        else:
            # Si déjà calculé avec ignore_far_sim=False ou s'il n'y a qu'une seule image sélectionnée
            reward = reward_div
    else:
        raise ValueError("reward_type doit être 'dr', 'd', 'r', ou 'd-nolambda'")

    return reward
