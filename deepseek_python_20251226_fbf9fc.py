# fix_attrs.py
with open('extract_features_cpu.py', 'r') as f:
    lines = f.readlines()

with open('extract_features_cpu.py', 'w') as f:
    for line in lines:
        if '.attrs[' in line and '=' in line:
            f.write('# ' + line)  # Commente la ligne
        else:
            f.write(line)
print("Fichier corrigé !")