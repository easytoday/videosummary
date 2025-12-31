# scripts/run_cpu_extraction.py
#!/usr/bin/env python3
"""
Script simplifié pour lancer l'extraction sur CPU
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*60)
    print("🚀 LANCEUR D'EXTRACTION CPU - GoogLeNet")
    print("="*60)
    
    # Configuration interactive
    video_folder = input("Dossier des vidéos [./videos]: ").strip() or "./videos"
    output_dir = input("Dossier de sortie [./dataset_cpu]: ").strip() or "./dataset_cpu"
    
    print("\n⚙️  Options d'extraction:")
    print("  1. Standard (FPS=1, batch=16)")
    print("  2. Rapide (FPS=0.5, batch=32) - moins de frames")
    print("  3. Qualité (FPS=2, batch=8) - plus lent")
    
    choice = input("\nChoisissez une option [1]: ").strip() or "1"
    
    if choice == "1":
        fps, batch = 1, 16
    elif choice == "2":
        fps, batch = 0.5, 32
    elif choice == "3":
        fps, batch = 2, 8
    else:
        fps, batch = 1, 16
    
    print(f"\n📊 Configuration sélectionnée:")
    print(f"   FPS: {fps}")
    print(f"   Batch size: {batch}")
    print(f"   Vidéos: {video_folder}")
    print(f"   Sortie: {output_dir}")
    
    # Estimation
    print("\n⏱️  Estimation du temps...")
    from extract_features_cpu import VideoDatasetProcessorCPU
    processor = VideoDatasetProcessorCPU(output_dir)
    processor.estimate_extraction_time(video_folder, fps)
    
    # Confirmation
    confirm = input("\n🚀 Lancer l'extraction? [O/n]: ").strip().lower()
    
    if confirm in ['', 'o', 'oui', 'y', 'yes']:
        print("\n🎬 Lancement de l'extraction...")
        
        # Construire la commande
        cmd = [
            sys.executable, "extract_features_cpu.py",
            "--video_folder", video_folder,
            "--output_dir", output_dir,
            "--fps", str(fps),
            "--batch_size", str(batch)
        ]
        
        # Ajouter l'option resume si demandé
        if Path(output_dir).exists() and any(Path(output_dir).glob("*.h5")):
            resume = input("📁 Un dataset existe déjà. Reprendre? [O/n]: ").strip().lower()
            if resume in ['', 'o', 'oui', 'y', 'yes']:
                cmd.append("--resume")
        
        print(f"\n💻 Commande exécutée:")
        print(f"   {' '.join(cmd)}")
        
        # Lancer dans un terminal séparé ou en arrière-plan
        print("\n📝 Les logs seront sauvegardés dans:")
        print(f"   - {output_dir}/extraction.log")
        print(f"   - console")
        
        input("\nAppuyez sur Entrée pour démarrer...")
        
        # Exécuter
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n⏸️  Extraction interrompue. Vous pouvez reprendre avec --resume")
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
    else:
        print("❌ Annulé.")

if __name__ == "__main__":
    main()
