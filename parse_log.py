import os
import argparse
import re
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

def movingaverage(values, window):
    if len(values) < window:
        return values
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="Chemin vers log_train.txt")
args = parser.parse_args()

if not osp.exists(args.path):
    raise ValueError(f"Chemin invalide : {args.path}")

# Regex pour capturer le nombre après 'reward'
regex_reward = re.compile(r'reward ([\.\deE+-]+)')
rewards = []

with open(args.path, 'r') as f:
    for line in f:
        reward_match = regex_reward.search(line)
        if reward_match:
            rewards.append(float(reward_match.group(1)))

if not rewards:
    print("Aucune donnée de reward trouvée. Vérifiez le format du log.")
else:
    rewards = np.array(rewards)
    smoothed_rewards = movingaverage(rewards, 8)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, color='blue', linewidth=2, label='Moyenne globale lissée')
    plt.fill_between(range(len(smoothed_rewards)), smoothed_rewards, alpha=0.1, color='blue')
    
    plt.xlabel('Époque / Étape')
    plt.ylabel('Récompense Moyenne')
    plt.title("Progression Globale de l'Apprentissage (RL)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_path = osp.join(osp.dirname(args.path), 'overall_reward_graph.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Graphique global sauvegardé : {save_path}")
