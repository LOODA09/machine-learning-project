import numpy as np
import pandas as pd
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# Deep learning imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# STEP 1: Load Dataset
# ============================================================
print("=" * 60)
print("  HOTEL BOOKING CANCELLATION - TRAINING PIPELINE")
print("=" * 60)

khaled_dataset = pd.read_csv("KHALED.csv")
print(f"\nDataset Shape: {khaled_dataset.shape}")

# ============================================================
# STEP 2: Rename Columns & Clean Data
# ============================================================
khaled_dataset.columns = khaled_dataset.columns.str.strip().str.replace(" ", "_").str.replace("-", "_").str.lower()
khaled_dataset = khaled_dataset.convert_dtypes()

# Parse date
khaled_dataset["date_of_reservation"] = pd.to_datetime(
    khaled_dataset["date_of_reservation"], format='%m/%d/%Y', errors="coerce"
)
khaled_dataset.dropna(axis=0, inplace=True)

# ============================================================
# STEP 3: Feature Engineering
# ============================================================
khaled_dataset["guest_count"] = khaled_dataset["number_of_adults"] + khaled_dataset["number_of_children"]
khaled_dataset["stay_duration"] = khaled_dataset["number_of_week_nights"] + khaled_dataset["number_of_weekend_nights"]

khaled_dataset["arrival_day"] = khaled_dataset["date_of_reservation"].dt.day_name()
khaled_dataset["arrival_month"] = khaled_dataset["date_of_reservation"].dt.month
khaled_dataset["arrival_year"] = khaled_dataset["date_of_reservation"].dt.year

khaled_dataset["cancel_history"] = khaled_dataset.apply(
    lambda row: 0 if row["repeated"] == 0 else row["p_c"] / (row["p_c"] + row["p_not_c"]) if (row["p_c"] + row["p_not_c"]) > 0 else 0,
    axis=1
)
khaled_dataset["new_guest_flag"] = 1 - khaled_dataset["repeated"]

def categorize_advance(days):
    if days <= 1: return 0
    elif days <= 7: return 1
    elif days <= 30: return 2
    elif days <= 365: return 3
    else: return 4

khaled_dataset["booking_advance"] = khaled_dataset["lead_time"].apply(categorize_advance)

def categorize_stay(nights):
    if nights == 0: return 0
    elif nights <= 3: return 1
    elif nights <= 7: return 2
    elif nights <= 14: return 3
    else: return 4

khaled_dataset["stay_duration"] = khaled_dataset["stay_duration"].apply(categorize_stay)

def group_guests(count):
    if count > 5: return "Group"
    return count

khaled_dataset["guest_count"] = khaled_dataset["guest_count"].apply(group_guests)

day_label_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
khaled_dataset["arrival_day"] = khaled_dataset["arrival_day"].map(day_label_map)
khaled_dataset["cancel_history"] = khaled_dataset["cancel_history"].round(2)

# ============================================================
# STEP 4: Drop Unnecessary Columns
# ============================================================
drop_features = [
    "booking_id", "number_of_adults", "number_of_children",
    "number_of_weekend_nights", "number_of_week_nights",
    "date_of_reservation", "p_c", "p_not_c", "repeated", "lead_time",
    "arrival_year", "arrival_month"  # Dropping to keep features matching Streamlit
]
khaled_dataset.drop(columns=drop_features, axis=1, inplace=True, errors='ignore')

# ============================================================
# STEP 5: Outlier Treatment
# ============================================================
room_rate_median = khaled_dataset["average_price"].median()
khaled_dataset.loc[khaled_dataset["average_price"] >= 179.5, "average_price"] = room_rate_median

# ============================================================
# STEP 6: One-Hot Encoding
# ============================================================
encode_columns = ["type_of_meal", "room_type", "market_segment_type", "booking_status", "guest_count"]
khaled_dataset = pd.get_dummies(khaled_dataset, columns=encode_columns, drop_first=True)

status_col = [c for c in khaled_dataset.columns if "booking_status" in c and "Not_Canceled" in c]
if status_col:
    khaled_dataset.rename(columns={status_col[0]: "reservation_outcome"}, inplace=True)

# ============================================================
# STEP 7: Split Features & Target
# ============================================================
feature_matrix = khaled_dataset.drop("reservation_outcome", axis=1)
target_vector = khaled_dataset["reservation_outcome"]

train_features, test_features, train_labels, test_labels = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42, stratify=target_vector
)

# Convert features and labels to float32 to fix deep learning dtype issues
train_features = train_features.astype(np.float32)
test_features = test_features.astype(np.float32)
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

# ============================================================
# STEP 8: SMOTE Balancing
# ============================================================
balance_sampler = SMOTE(random_state=42)
train_features, train_labels = balance_sampler.fit_resample(train_features, train_labels)

# ============================================================
# STEP 9: Feature Scaling
# ============================================================
feature_scaler = StandardScaler()
train_features["average_price"] = feature_scaler.fit_transform(train_features[["average_price"]])
test_features["average_price"] = feature_scaler.transform(test_features[["average_price"]])

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(feature_scaler, scaler_file)

# ============================================================
# STEP 10: K-Fold Cross Validation Setup & Evaluation
# ============================================================
kfold_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = {}

def evaluate_with_cv(model_instance, model_name, train_X, train_y, test_X, test_y, is_keras=False):
    print(f"\n{'='*50}\n  Training: {model_name}\n{'='*50}")
    
    t0 = time.time()
    if is_keras:
        model_instance.fit(train_X, train_y, epochs=30, batch_size=64, validation_split=0.1, verbose=0,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    else:
        model_instance.fit(train_X, train_y)
    train_time = time.time() - t0

    t1 = time.time()
    if is_keras:
        forecast_train = (model_instance.predict(train_X, verbose=0) > 0.5).astype(int).flatten()
        forecast_test = (model_instance.predict(test_X, verbose=0) > 0.5).astype(int).flatten()
    else:
        forecast_train = model_instance.predict(train_X)
        forecast_test = model_instance.predict(test_X)
    test_time = time.time() - t1

    train_accuracy = accuracy_score(train_y, forecast_train)
    test_accuracy = accuracy_score(test_y, forecast_test)
    test_precision = precision_score(test_y, forecast_test)
    test_recall = recall_score(test_y, forecast_test)
    test_f1 = f1_score(test_y, forecast_test)

    cv_mean = 0
    cv_std = 0
    if not is_keras and model_name != "SVM":
        cv_scores = cross_val_score(model_instance, train_X, train_y, cv=kfold_validator, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

    metrics = {
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_f1": round(test_f1, 4),
        "cv_mean": round(cv_mean, 4),
        "cv_std": round(cv_std, 4),
        "train_time_sec": round(train_time, 4),
        "test_time_sec": round(test_time, 4)
    }

    print(f"  Test Acc: {test_accuracy:.4f} | Time: {train_time:.2f}s")
    return metrics, model_instance

# ============================================================
# MODELS
# ============================================================
logistic_predictor = LogisticRegression(max_iter=1000, random_state=42)
all_metrics["Logistic Regression"], logistic_predictor = evaluate_with_cv(
    logistic_predictor, "Logistic Regression", train_features, train_labels, test_features, test_labels)

forest_predictor = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=5, class_weight='balanced', random_state=42, n_jobs=-1)
all_metrics["Random Forest"], forest_predictor = evaluate_with_cv(
    forest_predictor, "Random Forest", train_features, train_labels, test_features, test_labels)

knn_predictor = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
all_metrics["KNN"], knn_predictor = evaluate_with_cv(
    knn_predictor, "KNN", train_features, train_labels, test_features, test_labels)

xgb_predictor = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42, eval_metric='logloss', n_jobs=-1)
all_metrics["XGBoost"], xgb_predictor = evaluate_with_cv(
    xgb_predictor, "XGBoost", train_features, train_labels, test_features, test_labels)

svm_predictor = SVC(kernel='rbf', C=1.0, probability=True, random_state=42, max_iter=1000)
all_metrics["SVM"], svm_predictor = evaluate_with_cv(
    svm_predictor, "SVM", train_features, train_labels, test_features, test_labels)

tree_predictor = DecisionTreeClassifier(max_depth=20, random_state=42)
all_metrics["Decision Tree"], tree_predictor = evaluate_with_cv(
    tree_predictor, "Decision Tree", train_features, train_labels, test_features, test_labels)

# ============================================================
# K-Means
# ============================================================
print(f"\n{'='*50}\n  Training: K-Means Segmentation\n{'='*50}")
t0 = time.time()
kmeans_segmentor = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_assignments = kmeans_segmentor.fit_predict(train_features)
train_time = time.time() - t0

cluster_label_map = {}
for cluster_id in range(2):
    mask = cluster_assignments == cluster_id
    majority_label = train_labels[mask].mode()[0]
    cluster_label_map[cluster_id] = majority_label

t1 = time.time()
kmeans_predictions = np.array([cluster_label_map[c] for c in kmeans_segmentor.predict(test_features)])
test_time = time.time() - t1

all_metrics["K-Means"] = {
    "train_accuracy": 0.0, "test_accuracy": round(accuracy_score(test_labels, kmeans_predictions), 4),
    "test_precision": round(precision_score(test_labels, kmeans_predictions), 4),
    "test_recall": round(recall_score(test_labels, kmeans_predictions), 4),
    "test_f1": round(f1_score(test_labels, kmeans_predictions), 4),
    "cv_mean": 0, "cv_std": 0,
    "train_time_sec": round(train_time, 4), "test_time_sec": round(test_time, 4)
}

# ============================================================
# Deep Learning Models (Fixed Data Types)
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
all_metrics["ANN"], ann_model = evaluate_with_cv(
    ann_model, "ANN (Keras)", train_features, train_labels, test_features, test_labels, is_keras=True)

khaled_sequence_len = train_features.shape[1]
train_rnn_X = train_features.values.reshape(-1, 1, khaled_sequence_len)
test_rnn_X = test_features.values.reshape(-1, 1, khaled_sequence_len)
train_rnn_y = train_labels.values
test_rnn_y = test_labels.values

rnn_model = Sequential([
    Input(shape=(1, khaled_sequence_len)),
    SimpleRNN(units=128, activation='relu', return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
all_metrics["RNN"], rnn_model = evaluate_with_cv(
    rnn_model, "RNN", train_rnn_X, train_rnn_y, test_rnn_X, test_rnn_y, is_keras=True)

lstm_model = Sequential([
    Input(shape=(1, khaled_sequence_len)),
    LSTM(units=128, activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
all_metrics["LSTM"], lstm_model = evaluate_with_cv(
    lstm_model, "LSTM", train_rnn_X, train_rnn_y, test_rnn_X, test_rnn_y, is_keras=True)

# Save Keras models
save_model(ann_model, "ann_model.h5")
save_model(rnn_model, "rnn_model.h5")
save_model(lstm_model, "lstm_model.h5")

# ============================================================
# SHAP Values Computation
# ============================================================
print("\nComputing SHAP values for Random Forest...")
explainer = shap.TreeExplainer(forest_predictor)
sample_X = test_features.sample(min(500, test_features.shape[0]), random_state=42)
shap_values = explainer.shap_values(sample_X)
with open("shap_data.pkl", "wb") as f:
    pickle.dump({"explainer": explainer, "shap_values": shap_values, "sample_X": sample_X}, f)

# ============================================================
# Save Models & Configs
# ============================================================
all_sklearn_models = {
    "logistic_regression": logistic_predictor,
    "random_forest": forest_predictor,
    "knn": knn_predictor,
    "xgboost": xgb_predictor,
    "svm": svm_predictor,
    "decision_tree": tree_predictor,
    "kmeans": kmeans_segmentor
}

with open("model.pkl", "wb") as f:
    pickle.dump(all_sklearn_models, f)

model_config = {
    "feature_names": list(train_features.columns),
    "target_name": "reservation_outcome",
    "khaled_n_clusters": 2,
    "cluster_label_map": cluster_label_map,
    "rnn_sequence_len": khaled_sequence_len,
    "rnn_time_steps": 1
}

with open("model_config.pkl", "wb") as f:
    pickle.dump(model_config, f)

with open("training_metrics.pkl", "wb") as f:
    pickle.dump(all_metrics, f)

print("\nAll models and artifacts saved successfully.")
