import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="Chemin vers result.h5")
parser.add_argument('-d', '--frm-dir', type=str, required=True, help="Dossier des images (frames)")
parser.add_argument('-i', '--idx', type=int, default=0, help="Index de la vidéo dans le fichier h5")
parser.add_argument('--fps', type=int, default=30, help="Images par seconde")
parser.add_argument('--width', type=int, default=640, help="Largeur")
parser.add_argument('--height', type=int, default=480, help="Hauteur")
parser.add_argument('--save-dir', type=str, default='log', help="Dossier de sortie")
parser.add_argument('--save-name', type=str, default='summary.mp4', help="Nom du fichier MP4")
args = parser.parse_args()

def frm2video(frm_dir, summary, vid_writer):
    print(f"Assemblage de la vidéo... {int(sum(summary))} frames à traiter.")
    for idx, val in enumerate(summary):
        if val == 1:
            # Format standard du dataset : 000001.jpg, 000002.jpg...
            frm_name = str(idx+1).zfill(6) + '.jpg'
            frm_path = osp.join(frm_dir, frm_name)
            
            frm = cv2.imread(frm_path)
            if frm is not None:
                frm = cv2.resize(frm, (args.width, args.height))
                vid_writer.write(frm)
            else:
                # Optionnel : log si une frame manque
                continue

if __name__ == '__main__':
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Création de l'objet VideoWriter
    vid_writer = cv2.VideoWriter(
        osp.join(args.save_dir, args.save_name),
        cv2.VideoWriter_fourcc(*'mp4v'), # Codec compatible MP4
        args.fps,
        (args.width, args.height),
    )
    
    with h5py.File(args.path, 'r') as h5_res:
        # Correction pour Python 3 : conversion en liste pour l'indexation
        keys = list(h5_res.keys())
        key = keys[args.idx]
        print(f"Génération du résumé pour la vidéo : {key}")
        summary = h5_res[key]['machine_summary'][...]
        
    frm2video(args.frm_dir, summary, vid_writer)
    vid_writer.release()
    print(f"Terminé ! Vidéo sauvegardée dans {args.save_dir}/{args.save_name}")
