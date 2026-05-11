import json

def update_notebook():
    try:
        with open('hotel_cancellation_project(1).ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        changed = False
        for i, cell in enumerate(nb['cells']):
            src = ''.join(cell['source'])
            
            # Update ANN to use Keras
            if 'MLPClassifier' in src and 'ann_predictor' in src:
                new_src = '''# ============================================================
# CELL 22: MODEL 8 - ANN (Keras)
# ============================================================
ann_model = Sequential([
    Input(shape=(train_features.shape[1],)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_ann = ann_model.fit(
    train_features, train_labels,
    epochs=40, batch_size=64,
    validation_split=0.1, verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

ann_test_pred = (ann_model.predict(test_features, verbose=0) > 0.5).astype(int).flatten()
ann_test_acc = accuracy_score(test_labels, ann_test_pred)

ann_metrics = {
    "train_accuracy": max(history_ann.history['accuracy']),
    "test_accuracy": ann_test_acc,
    "test_precision": precision_score(test_labels, ann_test_pred),
    "test_recall": recall_score(test_labels, ann_test_pred),
    "test_f1": f1_score(test_labels, ann_test_pred),
    "cv_mean": 0,
    "cv_std": 0
}
all_metrics["ANN"] = ann_metrics

print(f"\\n  ANN Test Accuracy: {ann_test_acc:.4f}")
'''
                nb['cells'][i]['source'] = [line + '\n' for line in new_src.split('\n')]
                changed = True
                
            # Remove ann_predictor from model saving dict
            if '"ann": ann_predictor' in src:
                new_src = src.replace(',\\n    "ann": ann_predictor', '').replace(',\n    "ann": ann_predictor', '')
                nb['cells'][i]['source'] = [line + '\n' for line in new_src.split('\n')]
                changed = True

            # Update top_models
            if '"ANN": ann_predictor' in src:
                new_src = src.replace('    "ANN": ann_predictor\\n', '').replace('    "ANN": ann_predictor\n', '')
                nb['cells'][i]['source'] = [line + '\n' for line in new_src.split('\n')]
                changed = True

        if changed:
            with open('hotel_cancellation_project(1).ipynb', 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print("Successfully updated the notebook with Keras ANN and epochs.")
        else:
            print("No changes were needed. Keras ANN might already be configured.")
            
    except Exception as e:
        print(f"Error updating notebook: {e}")

if __name__ == "__main__":
    update_notebook()
