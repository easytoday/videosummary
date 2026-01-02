import json

json_path = 'datasets/alt_splits_kts.json'

# 1. Charger le JSON actuel
with open(json_path, 'r') as j:
    splits = json.load(j)

def format_key(k):
    # Transforme 'v9' en 'v00009'
    # On extrait le nombre après le 'v' et on le formate sur 5 digits
    num = int(k[1:])
    return f"v{num:05d}"

# 2. Transformer toutes les clés dans les 5 splits
new_splits = []
for split in splits:
    new_splits.append({
        "train_keys": [format_key(k) for k in split['train_keys']],
        "test_keys": [format_key(k) for k in split['test_keys']]
    })

# 3. Sauvegarder le JSON corrigé
with open(json_path, 'w') as j:
    json.dump(new_splits, j, indent=4)

print("Succès : Toutes les clés du JSON ont été converties au format vXXXXX (5 digits).")
