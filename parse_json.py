import os
import argparse
import os.path as osp
import matplotlib
matplotlib.use('Agg') # Pour générer l'image sans interface graphique
from matplotlib import pyplot as plt
from utils import read_json
import numpy as np

def movingaverage(values, window):
    if len(values) < window:
        return values
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="Chemin vers rewards.json")
parser.add_argument('-i', '--idx', type=int, default=0, help="Index de la vidéo à visualiser")
args = parser.parse_args()

# Chargement des données
reward_writers = read_json(args.path)
keys = list(reward_writers.keys()) # Conversion en liste pour Python 3
assert args.idx < len(keys), f"L'index {args.idx} dépasse le nombre de vidéos ({len(keys)})"

key = keys[args.idx]
rewards = reward_writers[key]

# Calcul et lissage
rewards = np.array(rewards)
smoothed_rewards = movingaverage(rewards, 8)

# Génération du graphique
plt.figure(figsize=(10, 5))
plt.plot(smoothed_rewards, label='Récompense lissée (MA=8)')
plt.plot(rewards, alpha=0.3, label='Récompense brute', color='gray') # On garde le brut en fond
plt.xlabel('Épisode / Époque')
plt.ylabel('Récompense (Reward)')
plt.title(f"Évolution de l'apprentissage - Vidéo: {key}")
plt.legend()
plt.grid(True)

# Sauvegarde
output_path = osp.join(osp.dirname(args.path), f'learning_curve_{key}.png')
plt.savefig(output_path)
plt.close()

print(f"Graphique sauvegardé sous : {output_path}")
