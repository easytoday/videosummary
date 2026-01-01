import h5py

with h5py.File('log/alt_run_kts/result.h5', 'r') as f:
    for i, key in enumerate(f.keys()):
        summary = f[key]['machine_summary'][()]
        n_selected = sum(summary)
        total = len(summary)
        ratio = (n_selected / total) * 100
        print(f"Index {i} | Vidéo: {key} | Sélection: {n_selected}/{total} frames ({ratio:.2f}%)")
