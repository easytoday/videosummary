import h5py
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_PATH = "dataset_cpu/features.h5"       # Votre fichier actuel
OUTPUT_PATH = "dataset_cpu/alt_dataset.h5"   # Le fichier corrigé pour Zhou
SAMPLING_RATE = 15  # On suppose 1 feature toutes les 15 frames (standard)

def restructure_h5():
    if not os.path.exists(INPUT_PATH):
        print(f"Erreur : Le fichier {INPUT_PATH} n'existe pas.")
        return

    print(f"Lecture de {INPUT_PATH}...")
    print(f"Écriture vers {OUTPUT_PATH}...")

    with h5py.File(INPUT_PATH, 'r') as f_in, h5py.File(OUTPUT_PATH, 'w') as f_out:
        keys = list(f_in.keys())
        total = len(keys)
        
        for i, key in enumerate(keys):
            # 1. Récupération des features brutes
            # Attention : on gère le cas où v1 est un Groupe ou un Dataset direct
            input_item = f_in[key]
            
            if isinstance(input_item, h5py.Group):
                # Si c'est déjà un groupe mais mal formé, on essaie de trouver 'features'
                if 'features' in input_item:
                    features = input_item['features'][...]
                else:
                    print(f"⚠️ Ignoré {key} : structure de groupe inconnue")
                    continue
            else:
                # C'est le cas probable : v1 est directement le dataset
                features = input_item[...]

            # Vérification dimensionnelle (1024 attendu)
            # Si vous avez du 2048 (ResNet), on garde tel quel, 
            # mais il faudra changer --input-dim dans main.py
            n_feats, dim = features.shape
            
            # 2. Création de la structure correcte (Le "Tiroir")
            group = f_out.create_group(key)
            
            # --- A. FEATURES ---
            group.create_dataset('features', data=features)
            
            # --- B. PICKS (Indices des frames) ---
            # On recrée les positions temporelles artificiellement
            # 0, 15, 30, 45...
            picks = np.arange(0, n_feats * SAMPLING_RATE, SAMPLING_RATE)
            group.create_dataset('picks', data=picks)
            
            # --- C. N_FRAMES ---
            n_frames = n_feats * SAMPLING_RATE
            group.create_dataset('n_frames', data=n_frames)
            
            # --- D. CHANGE_POINTS (Segmentation) ---
            # C'est ici que ça bloquait. Sans l'algo KTS complexe, 
            # on découpe la vidéo en segments réguliers de ~5 secondes (75 frames = 5 features)
            # pour débloquer l'entraînement.
            segment_len = 5 # features
            n_segments = int(np.ceil(n_feats / segment_len))
            
            change_points = []
            n_frame_per_seg = []
            
            for j in range(n_segments):
                start_feat = j * segment_len
                end_feat = min((j + 1) * segment_len, n_feats)
                
                # Conversion en indices de frames
                start_frame = start_feat * SAMPLING_RATE
                # La fin du segment est la dernière frame du bloc
                end_frame = (end_feat * SAMPLING_RATE) - 1
                
                change_points.append([start_frame, end_frame])
                n_frame_per_seg.append(end_frame - start_frame + 1)
            
            group.create_dataset('change_points', data=np.array(change_points))
            group.create_dataset('n_frame_per_seg', data=np.array(n_frame_per_seg))
            
            # --- E. GT SCORE (Optionnel mais évite les crashs) ---
            # On met des zéros car on n'a pas encore de vérité terrain humaine
            group.create_dataset('gtscore', data=np.zeros(n_frames))
            group.create_dataset('user_summary', data=np.zeros((1, n_frames)))

            if i % 10 == 0:
                print(f"Traité {i}/{total} vidéos...")

    print("\n✅ Terminé ! Nouveau dataset généré : dataset_cpu/alt_dataset.h5")
    print("Utilisez ce fichier pour create_split.py et main.py")

if __name__ == "__main__":
    restructure_h5()
