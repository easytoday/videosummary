import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="Chemin vers result.h5 (généré par main.py)")
parser.add_argument('-v', '--video-path', type=str, required=True, help="Chemin vers le fichier .mp4 original")
parser.add_argument('-i', '--idx', type=int, default=0, help="Index de la vidéo dans le fichier h5")
parser.add_argument('--save-dir', type=str, default='summaries', help="Dossier de sortie")
args = parser.parse_args()

def generate_summary():
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 1. Ouvrir le fichier de résultats H5
    with h5py.File(args.path, 'r') as h5_res:
        keys = list(h5_res.keys())
        key = keys[args.idx]
        print(f"Génération du résumé pour : {key}")
        
        # Le machine_summary est un vecteur binaire (0 ou 1 pour chaque frame)
        summary_mask = h5_res[key]['machine_summary'][...]

    # 2. Ouvrir la vidéo originale pour récupérer les propriétés
    cap = cv2.VideoCapture(args.video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # 3. Préparer le VideoWriter
    save_name = f"summary_{key}.mp4"
    out_path = osp.join(args.save_dir, save_name)
    vid_writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # 4. Parcourir la vidéo et écrire uniquement les frames sélectionnées
    print(f"Extraction des segments... ({int(sum(summary_mask))} frames)")
    
    frame_idx = 0
    pbar = tqdm(total=len(summary_mask))
    
    while True:
        success, frame = cap.read()
        if not success or frame_idx >= len(summary_mask):
            break
        
        # Si le masque renvoie 1, on garde la frame
        if summary_mask[frame_idx] == 1:
            vid_writer.write(frame)
        
        frame_idx += 1
        pbar.update(1)

    cap.release()
    vid_writer.release()
    pbar.close()
    print(f"\nRésumé sauvegardé : {out_path}")

if __name__ == '__main__':
    generate_summary()
