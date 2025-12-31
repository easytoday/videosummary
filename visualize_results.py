import h5py
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True,
                    help="Chemin vers le fichier result.h5")
args = parser.parse_args()

if not osp.exists(args.path):
    print(f"Erreur : Le fichier {args.path} est introuvable.")
    exit()

h5_res = h5py.File(args.path, 'r')
keys = list(h5_res.keys())

print(f"Analyse de {len(keys)} vidéos...")

for key in keys:
    # Récupération des données
    score = h5_res[key]['score'][...]
    machine_summary = h5_res[key]['machine_summary'][...]
    gtscore = h5_res[key]['gtscore'][...]
    fm = h5_res[key]['fm'][()]

    # Création de la figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    n = len(gtscore)
    
    # Graphique du haut : Ground Truth (Rouge)
    axs[0].plot(range(n), gtscore, color='red', label='Humains (Ground Truth)')
    axs[0].set_xlim(0, n)
    axs[0].set_ylabel('GT Score')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[0].legend(loc='upper right')
    axs[0].set_title(f"Vidéo : {key} | F-score : {fm:.1%}")

    # Graphique du bas : Prédiction Machine (Bleu)
    # Note : Le score peut avoir une longueur différente du GT à cause du sous-échantillonnage,
    # on adapte l'axe X pour que la comparaison soit alignée.
    n_score = len(score)
    axs[1].plot(range(n_score), score, color='blue', label='IA (DSN Prediction)')
    axs[1].set_xlim(0, n_score)
    axs[1].set_ylabel('IA Score')
    axs[1].set_yticklabels([])
    axs[1].legend(loc='upper right')

    # Sauvegarde
    save_name = osp.join(osp.dirname(args.path), f'compare_{key}.png')
    fig.savefig(save_name, bbox_inches='tight')
    plt.close(fig)

    print(f"Terminé pour {key}. # frames {len(machine_summary)}. Image : {save_name}")

h5_res.close()
