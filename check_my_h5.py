import h5py
import numpy as np

# Chemin vers votre fichier
#FILE_PATH = "dataset_cpu/features.h5" 
FILE_PATH = "dataset_cpu/alt_dataset.h5" # Aziz après lancement restructure_dataset.py 

def check_video_structure(path):
    try:
        f = h5py.File(path, 'r')
    except OSError:
        print(f"❌ ERREUR CRITIQUE : Impossible d'ouvrir {path}. Le fichier est corrompu ou le chemin est faux.")
        return

    print(f"📂 Analyse de : {path}")
    print(f"   Nombre de vidéos trouvées : {len(f.keys())}")
    
    if len(f.keys()) == 0:
        print("❌ ERREUR : Le fichier est vide.")
        return

    # On prend la première vidéo pour tester
    first_key = list(f.keys())[0]
    data = f[first_key]
    print(f"   Test sur la vidéo : '{first_key}'")

    # 1. Vérification des Features
    if 'features' not in data:
        print("❌ MANQUANT : Pas de clé 'features'.")
    else:
        feat_shape = data['features'].shape
        print(f"   ✅ Features détectées. Forme : {feat_shape}")
        if feat_shape[1] != 1024:
            print(f"   ⚠️ ATTENTION : Dimension = {feat_shape[1]}. Le modèle attend 1024. "
                  f"Il faudra modifier args.input_dim dans main.py ou projeter les données.")

    # 2. Vérification des Change Points (Crucial pour Zhou)
    if 'change_points' not in data:
        print("❌ MANQUANT : Pas de clé 'change_points'.")
        print("   -> SOLUTION : Vous devez exécuter un algorithme de détection de plans (KTS) sur vos features.")
    else:
        cp_shape = data['change_points'].shape
        print(f"   ✅ Change Points détectés. Forme : {cp_shape} (Doit être N x 2)")

    # 3. Vérification du n_frame_per_seg
    if 'n_frame_per_seg' not in data:
        print("❌ MANQUANT : Pas de clé 'n_frame_per_seg'.")
    else:
        print("   ✅ n_frame_per_seg présent.")

    # 4. Vérification des Picks (Frames sous-échantillonnées)
    if 'picks' not in data:
        print("❌ MANQUANT : Pas de clé 'picks'.")
        print("   -> Le code ne saura pas retrouver les frames originales pour summary2video.py.")
    else:
        print(f"   ✅ Picks présents. ({len(data['picks'])} indices)")

    f.close()

if __name__ == "__main__":
    check_video_structure(FILE_PATH)
