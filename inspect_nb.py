import json

with open('hotel_cancellation_project(1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# Show full source of each cell
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    ct = c['cell_type']
    print(f"\n{'='*60}")
    print(f"CELL {i} [{ct}]")
    print(f"{'='*60}")
    print(src[:500])
    if len(src) > 500:
        print(f"... ({len(src)} total chars)")
