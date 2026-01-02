import h5py
import numpy as np
import shutil
import os
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Analyse automatique des pics d'importance.")
    parser.add_argument('input_dir', type=str, help="Chemin du dossier (ex: videos/v00072_frames/)")
    parser.add_argument('-n', '--num', type=int, default=5, help="Nombre de pics (défaut: 5)")
    args = parser.parse_args()

    # 1. Extraire l'ID de la vidéo du chemin (ex: v72 ou v00072)
    # On cherche 'v' suivi de chiffres
    match = re.search(r'v(\d+)', args.input_dir)
    if not match:
        print(f"Erreur : Impossible de trouver l'ID de la vidéo dans le chemin {args.input_dir}")
        return
    
    video_id = f"v{int(match.group(1))}" # Normalise 'v00072' en 'v72' pour correspondre au H5
    h5_path = 'log/alt_run_kts/result.h5'

    if not os.path.exists(h5_path):
        print(f"Erreur : Fichier {h5_path} introuvable.")
        return

    # 2. Chercher l'index automatiquement dans le fichier H5
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        try:
            video_index = keys.index(video_id)
        except ValueError:
            print(f"Erreur : La vidéo {video_id} n'est pas dans le fichier de résultats.")
            print(f"Clés disponibles : {keys}")
            return

        print(f"ID détecté : {video_id} | Index H5 : {video_index}")
        
        scores = f[video_id]['score'][()]
        output_dir = f"analysis_peaks_{video_id}"
        os.makedirs(output_dir, exist_ok=True)

        # 3. Extraction des pics
        top_indices = np.argsort(scores)[-args.num:][::-1]

        print(f"--- Extraction des {args.num} pics principaux ---")
        for rank, idx in enumerate(top_indices):
            score_val = scores[idx]
            # On calcule le numéro de frame (subsample de 15 par défaut dans Zhou)
            real_frame_num = idx * 15
            
            # Formatage 000001.jpg
            frame_name = f"{real_frame_num:06d}.jpg"
            src_path = os.path.join(args.input_dir, frame_name)
            
            if os.path.exists(src_path):
                dest_name = f"rank{rank+1}_score{score_val:.2f}_frame{real_frame_num}.jpg"
                shutil.copy(src_path, os.path.join(output_dir, dest_name))
                print(f"Pic {rank+1} : Frame {real_frame_num} (Score: {score_val:.4f}) -> Copié.")
            else:
                # Tentative alternative si les frames commencent à 1 au lieu de 0
                frame_name_alt = f"{real_frame_num+1:06d}.jpg"
                src_path_alt = os.path.join(args.input_dir, frame_name_alt)
                if os.path.exists(src_path_alt):
                    shutil.copy(src_path_alt, os.path.join(output_dir, dest_name))
                    print(f"Pic {rank+1} : Frame {real_frame_num+1} (Score: {score_val:.4f}) -> Copié.")

if __name__ == "__main__":
    main()
