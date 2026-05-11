import json

with open('final project tam .ipynb', 'r', encoding='utf-8') as f:
    tam_nb = json.load(f)

with open('hotel_cancellation_project(1).ipynb', 'r', encoding='utf-8') as f:
    target_nb = json.load(f)

# The tam_nb has EDA in cells 5 to 50 approximately.
# I want to extract the plotly visualisations and markdown from tam_nb and insert them into target_nb.
eda_cells = []
for cell in tam_nb['cells']:
    src = ''.join(cell['source'])
    if 'px.' in src or 'sns.' in src or 'plt.' in src or cell['cell_type'] == 'markdown':
        if len(src) > 5 and not 'StandardScaler' in src and not 'train_test_split' in src:
            eda_cells.append(cell)

# Insert after cell 10 (which is typically after data cleaning in target_nb)
target_nb['cells'] = target_nb['cells'][:10] + eda_cells[:20] + target_nb['cells'][10:]

with open('hotel_cancellation_project(1).ipynb', 'w', encoding='utf-8') as f:
    json.dump(target_nb, f, indent=1)
print("Updated notebook with EDA cells")
