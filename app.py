# ===============================================
# EEG Emotion Classification Web App (Streamlit)
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Streamlit setup
st.set_page_config(page_title="EEG Emotion Classifier", layout="wide")
st.title("🧠 EEG Emotion Classification Web App")
st.write("Upload EEG data, select features and classifier, then visualize results interactively.")

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader("📤 Upload EEG CSV or XLSX File", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read uploaded file
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Display data preview
        st.dataframe(df.head())

        # ===========================================
        # Feature & Label Selection
        # ===========================================
        st.sidebar.header("Feature & Model Settings")
        label_col = st.sidebar.text_input("Label Column Name (e.g., Emotion)", "Emotion")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.sidebar.multiselect("Select EEG Features", numeric_cols, default=numeric_cols)
        classifier_name = st.sidebar.selectbox(
            "Select Classifier", ["KNN", "SVM", "Random Forest", "Neural Network"]
        )

        if st.sidebar.button("🚀 Run Classification"):
            try:
                if label_col not in df.columns:
                    st.error(f"Label column '{label_col}' not found.")
                else:
                    X = df[selected_features]
                    y = df[label_col]

                    # Encode labels
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # Scale
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # Choose classifier and param grid
                    if classifier_name == "KNN":
                        model = KNeighborsClassifier()
                        param_grid = {"n_neighbors": [3, 5, 7, 9]}
                    elif classifier_name == "SVM":
                        model = SVC()
                        param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                    elif classifier_name == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7, None]}
                    else:
                        model = MLPClassifier(max_iter=1000, random_state=42)
                        param_grid = {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "activation": ["relu", "tanh"]}

                    grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_

                    y_pred = best_model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    st.subheader("🏆 Model Performance Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                        "Score": [acc, prec, rec, f1]
                    })
                    st.table(metrics_df)

                    st.write(f"**Best Parameters:** {grid.best_params_}")

                    # ===== Grid Search Plot =====
                    st.subheader("📈 Grid Search Accuracy Plot")
                    grid_df = pd.DataFrame(grid.cv_results_)
                    plt.figure(figsize=(6, 4))
                    plt.plot(grid_df["mean_test_score"], marker='o')
                    plt.title(f"{classifier_name} Mean CV Accuracy")
                    plt.xlabel("Parameter Combination")
                    plt.ylabel("Mean CV Accuracy")
                    st.pyplot(plt)

                    # ===== Confusion Matrix =====
                    st.subheader("🔹 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=le.classes_, yticklabels=le.classes_)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(plt)

                    # ===== Feature Importance (RF only) =====
                    if classifier_name == "Random Forest":
                        st.subheader("🌟 Feature Importance")
                        importances = best_model.feature_importances_
                        sorted_idx = np.argsort(importances)[::-1]
                        plt.figure(figsize=(8, 4))
                        plt.bar(range(len(importances)), importances[sorted_idx])
                        plt.xticks(range(len(importances)), np.array(selected_features)[sorted_idx], rotation=90)
                        plt.tight_layout()
                        st.pyplot(plt)

            except Exception as e:
                st.error(f"❌ Error: {e}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Please upload a CSV or XLSX EEG dataset to start.")
