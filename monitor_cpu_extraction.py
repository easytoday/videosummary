# scripts/monitor_cpu_extraction.py
#!/usr/bin/env python3
"""
Script de monitoring pour suivre l'extraction sur CPU
"""

import time
import psutil
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class ExtractionMonitor:
    """Moniteur de l'extraction sur CPU"""
    
    def __init__(self, log_file='extraction.log', checkpoint_file='checkpoint.json'):
        self.log_file = Path(log_file)
        self.checkpoint_file = Path(checkpoint_file)
        
    def monitor_extraction(self, interval_seconds=30):
        """
        Surveille l'extraction en temps réel
        """
        print("🔍 Monitoring de l'extraction CPU...")
        print("Appuyez sur Ctrl+C pour arrêter le monitoring")
        
        metrics = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'processed_videos': []
        }
        
        try:
            while True:
                # Mesurer les métriques système
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory().percent
                
                timestamp = datetime.now().isoformat()
                
                metrics['timestamps'].append(timestamp)
                metrics['cpu_percent'].append(cpu)
                metrics['memory_percent'].append(memory)
                
                # Vérifier l'avancement
                processed_count = 0
                if self.checkpoint_file.exists():
                    with open(self.checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)
                        processed_count = len(checkpoint.get('processed_videos', []))
                
                metrics['processed_videos'].append(processed_count)
                
                # Afficher
                print(f"\r⏱️  {timestamp.split('T')[1][:8]} | "
                      f"CPU: {cpu:3.0f}% | "
                      f"Mémoire: {memory:3.0f}% | "
                      f"Vidéos traitées: {processed_count}", end='')
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n📈 Génération du rapport de monitoring...")
            self.generate_report(metrics)
    
    def generate_report(self, metrics):
        """Génère un rapport de monitoring"""
        if not metrics['timestamps']:
            print("❌ Aucune donnée à afficher")
            return
        
        # Convertir les timestamps
        timestamps = [datetime.fromisoformat(ts) for ts in metrics['timestamps']]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Utilisation CPU
        axes[0, 0].plot(timestamps, metrics['cpu_percent'], 'r-', linewidth=2)
        axes[0, 0].set_title('Utilisation CPU (%)')
        axes[0, 0].set_ylabel('Pourcentage')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Utilisation mémoire
        axes[0, 1].plot(timestamps, metrics['memory_percent'], 'b-', linewidth=2)
        axes[0, 1].set_title('Utilisation Mémoire (%)')
        axes[0, 1].set_ylabel('Pourcentage')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Vidéos traitées
        axes[1, 0].plot(timestamps, metrics['processed_videos'], 'g-', linewidth=2)
        axes[1, 0].set_title('Vidéos Traitées')
        axes[1, 0].set_ylabel('Nombre')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Statistiques
        axes[1, 1].axis('off')
        
        stats_text = f"""
        📊 Statistiques Finales:
        
        Durée monitoring: {len(timestamps)} échantillons
        CPU moyen: {np.mean(metrics['cpu_percent']):.1f}%
        Mémoire moyenne: {np.mean(metrics['memory_percent']):.1f}%
        
        Vidéos démarrées: {metrics['processed_videos'][0]}
        Vidéos terminées: {metrics['processed_videos'][-1]}
        Progression: {metrics['processed_videos'][-1] - metrics['processed_videos'][0]}
        
        ⏱️  Dernière mise à jour: {timestamps[-1].strftime('%H:%M:%S')}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('extraction_monitoring.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Rapport sauvegardé: extraction_monitoring.png")

if __name__ == "__main__":
    monitor = ExtractionMonitor()
    monitor.monitor_extraction(interval_seconds=30)