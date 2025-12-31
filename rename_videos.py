# rename_simple.py
import os
from pathlib import Path

def rename_to_simple(folder_path):
    folder = Path(folder_path)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    # Récupérer la liste des vidéos
    videos = sorted([f for f in folder.iterdir() if f.suffix.lower() in video_extensions])
    
    if not videos:
        print("Aucune vidéo trouvée.")
        return

    print(f"🔄 Renommage de {len(videos)} vidéos...")
    
    # Création d'un fichier de correspondance pour ne pas perdre l'info originale
    with open(folder / "mapping.txt", "w", encoding="utf-8") as f:
        f.write("Nouveau_Nom | Nom_Original\n")
        f.write("-" * 30 + "\n")
        
        for i, video_path in enumerate(videos, 1):
            ext = video_path.suffix
            new_name = f"v{i}{ext}"
            new_path = video_path.with_name(new_name)
            
            # Sauvegarde de la correspondance
            f.write(f"{new_name} | {video_path.name}\n")
            
            # Renommage effectif
            video_path.rename(new_path)
            print(f"✅ {video_path.name} -> {new_name}")

    print(f"\n✨ Terminé. Consultez '{folder}/mapping.txt' pour la correspondance.")

if __name__ == "__main__":
    rename_to_simple("videos")
