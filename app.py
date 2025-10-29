# ===============================================
# EEG Multi-Label Classification (Streamlit)
# ===============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.inspection import permutation_importance

# Streamlit setup
st.set_page_config(page_title="EEG Multi-Label Classifier", layout="wide")
st.title("🧠 EEG Multi-Label EEG Classification")
st.write("Upload your EEG dataset, choose multiple label columns, select classifier, and view training performance and feature importances.")

# ===============================================
# File Upload
# ===============================================
uploaded_file = st.file_uploader("📤 Upload EEG CSV or XLSX File", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} × {df.shape[1]}")
        st.dataframe(df.head())

        # ===============================================
        # Sidebar - Settings
        # ===============================================
        st.sidebar.header("⚙️ Model Settings")

        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Multi-label column selection
        label_cols = st.sidebar.multiselect(
            "Select Label Column(s)",
            [col for col in all_columns if col not in numeric_cols],
            help="You can select multiple label columns for multi-label classification."
        )

        # Feature selection
        default_features = numeric_cols.copy()
        selected_features = st.sidebar.multiselect(
            "Select EEG Features", numeric_cols, default=default_features
        )

        classifier_name = st.sidebar.selectbox(
            "Select Classifier", ["KNN", "SVM", "Random Forest", "Neural Network"]
        )

        test_size = st.sidebar.slider("Test Size (Ratio for Testing)", 0.1, 0.95, 0.2, step=0.05)

        run_button = st.sidebar.button("🚀 Run Classification")

        if run_button:
            if len(label_cols) == 0:
                st.warning("⚠️ Please select at least one label column.")
                st.stop()

            try:
                # ================== Data Preparation ==================
                X = df[selected_features]
                y = df[label_cols].copy()

                # Encode each label column separately
                label_encoders = {}
                for col in label_cols:
                    le = LabelEncoder()
                    y[col] = le.fit_transform(y[col].astype(str))
                    label_encoders[col] = le

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # ================== Model Selection ==================
                if classifier_name == "KNN":
                    base_model = KNeighborsClassifier()
                    param_grid = {"estimator__n_neighbors": [3, 5, 7, 9]}
                elif classifier_name == "SVM":
                    base_model = SVC(probability=True)
                    param_grid = {"estimator__C": [0.1, 1, 10], "estimator__kernel": ["linear", "rbf"]}
                elif classifier_name == "Random Forest":
                    base_model = RandomForestClassifier(random_state=42)
                    param_grid = {"estimator__n_estimators": [50, 100, 200], "estimator__max_depth": [3, 5, 7, None]}
                else:
                    base_model = MLPClassifier(max_iter=1000, random_state=42)
                    param_grid = {"estimator__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                                  "estimator__activation": ["relu", "tanh"]}

                # Multi-output wrapper for multiple labels
                model = MultiOutputClassifier(base_model)

                # Grid search across all outputs
                st.info("⏳ Running Grid Search... please wait...")
                grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, return_train_score=True)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

                # ================== Performance ==================
                y_pred = pd.DataFrame(best_model.predict(X_test), columns=y.columns)

                results = []
                for col in y.columns:
                    acc = accuracy_score(y_test[col], y_pred[col])
                    prec = precision_score(y_test[col], y_pred[col], average="weighted", zero_division=0)
                    rec = recall_score(y_test[col], y_pred[col], average="weighted", zero_division=0)
                    f1 = f1_score(y_test[col], y_pred[col], average="weighted", zero_division=0)
                    results.append([col, acc, prec, rec, f1])

                metrics_df = pd.DataFrame(results, columns=["Label", "Accuracy", "Precision", "Recall", "F1-Score"])
                st.subheader("🏆 Model Performance (Per Label)")
                st.dataframe(metrics_df.style.format("{:.3f}"))

                st.write(f"**Best Parameters:** {grid.best_params_}")

                # ================== Accuracy Comparison ==================
                st.subheader("📊 Average Performance Across Labels")
                avg_scores = metrics_df[["Accuracy", "Precision", "Recall", "F1-Score"]].mean().to_dict()
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=list(avg_scores.keys()), y=list(avg_scores.values()), palette="viridis", ax=ax)
                for i, v in enumerate(avg_scores.values()):
                    ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
                ax.set_ylim(0, 1)
                st.pyplot(fig)

                # ================== Feature Importance ==================
                st.subheader("🌟 Feature Importance (Permutation-based)")

                try:
                    base = best_model.estimators_[0]
                    imp_result = permutation_importance(base, X_test, y_test.iloc[:, 0], n_repeats=10)
                    importances = imp_result.importances_mean
                    sorted_idx = np.argsort(importances)[::-1]
                    plt.figure(figsize=(8, 4))
                    plt.bar(range(len(importances)), importances[sorted_idx])
                    plt.xticks(range(len(importances)), np.array(selected_features)[sorted_idx], rotation=90)
                    plt.title(f"{classifier_name} Feature Importance (Label: {y.columns[0]})")
                    plt.tight_layout()
                    st.pyplot(plt)
                except Exception as e:
                    st.warning(f"⚠️ Feature importance unavailable: {e}")

            except Exception as e:
                st.error(f"❌ Error during classification: {e}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Please upload a CSV or XLSX EEG dataset to begin.")
