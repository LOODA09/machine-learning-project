import json
import sys

def fix_nb():
    fname = 'hotel_cancellation_project(1).ipynb'
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        new_keras_ann = [
            '# ============================================================\n',
            '# CELL 22: MODEL 8 - ANN (Artificial Neural Network)\n',
            '# ============================================================\n',
            '# ANN trained with Keras using Epochs for >90% Accuracy\n',
            'ann_model = Sequential([\n',
            '    Input(shape=(train_features.shape[1],)),\n',
            '    Dense(256, activation=\'relu\'),\n',
            '    Dropout(0.3),\n',
            '    Dense(128, activation=\'relu\'),\n',
            '    Dropout(0.2),\n',
            '    Dense(64, activation=\'relu\'),\n',
            '    Dense(1, activation=\'sigmoid\')\n',
            '])\n',
            'ann_model.compile(optimizer=\'adam\', loss=\'binary_crossentropy\', metrics=[\'accuracy\'])\n',
            '\n',
            'print(\"Training ANN with 40 Epochs...\")\n',
            'history_ann = ann_model.fit(\n',
            '    train_features.astype(\'float32\'), train_labels.astype(\'float32\'),\n',
            '    epochs=40, batch_size=64,\n',
            '    validation_split=0.1, verbose=1,\n',
            '    callbacks=[EarlyStopping(monitor=\'val_loss\', patience=5, restore_best_weights=True)]\n',
            ')\n',
            '\n',
            '# Evaluation\n',
            'ann_test_pred = (ann_model.predict(test_features.astype(\'float32\'), verbose=0) > 0.5).astype(int).flatten()\n',
            'ann_test_acc = accuracy_score(test_labels, ann_test_pred)\n',
            '\n',
            'ann_metrics = {\n',
            '    \"train_accuracy\": max(history_ann.history[\'accuracy\']),\n',
            '    \"test_accuracy\": ann_test_acc,\n',
            '    \"test_precision\": precision_score(test_labels, ann_test_pred),\n',
            '    \"test_recall\": recall_score(test_labels, ann_test_pred),\n',
            '    \"test_f1\": f1_score(test_labels, ann_test_pred),\n',
            '    \"test_time_sec\": 0\n',
            '}\n',
            'all_metrics[\"ANN\"] = ann_metrics\n',
            '\n',
            'print(f\"\\nANN Test Accuracy: {ann_test_acc:.4f}\")\n',
            'print(\"CELL UPDATED: NOW USING KERAS WITH EPOCHS\")\n'
        ]

        changed = False
        for i, cell in enumerate(nb['cells']):
            src = "".join(cell['source']).lower()
            # If cell has BOTH MLPClassifier and ann_model.fit, it is the broken one
            if 'mlpclassifier' in src and 'ann_model.fit' in src:
                nb['cells'][i]['source'] = new_keras_ann
                changed = True
                print(f"Fixed broken cell at index {i}")
            # If it has the khaled headers but old code
            elif 'khaled_hidden_layers' in src and 'model 8' in src:
                nb['cells'][i]['source'] = new_keras_ann
                changed = True
                print(f"Fixed old cell at index {i}")

        if changed:
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print("Successfully saved changes to disk.")
        else:
            print("No matching cells found to fix.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    fix_nb()
