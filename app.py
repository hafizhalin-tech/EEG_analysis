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
from sklearn.inspection import permutation_importance

# Streamlit setup
st.set_page_config(page_title="EEG Emotion Classifier", layout="wide")
st.title("🧠 EEG Emotion Classification Web App")
st.write("Upload EEG data, select features and classifier, then visualize results interactively.")

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader("📤 Upload EEG CSV or XLSX File", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Display data preview
        st.dataframe(df.head())

        # ===========================================
        # Sidebar - Settings
        # ===========================================
        st.sidebar.header("⚙️ Model Settings")

        label_col = st.sidebar.text_input("Label Column Name (e.g., Emotion)", "Emotion")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.sidebar.multiselect("Select EEG Features", numeric_cols, default=numeric_cols)
        classifier_name = st.sidebar.selectbox("Select Classifier", ["KNN", "SVM", "Random Forest", "Neural Network"])
        test_size = st.sidebar.slider("Test Size (Ratio of data for testing)", 0.1, 0.95, 0.2, step=0.05)

        run_button = st.sidebar.button("🚀 Run Classification")

        if run_button:
            if label_col not in df.columns:
                st.error(f"Label column '{label_col}' not found.")
            else:
                try:
                    # ======== Data Preparation ========
                    X = df[selected_features]
                    y = df[label_col]

                    # Encode labels
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    # Scale
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # ======== Model & Parameter Grid ========
                    if classifier_name == "KNN":
                        model = KNeighborsClassifier()
                        param_grid = {"n_neighbors": [3, 5, 7, 9]}
                    elif classifier_name == "SVM":
                        model = SVC(probability=True)
                        param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                    elif classifier_name == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7, None]}
                    else:
                        model = MLPClassifier(max_iter=1000, random_state=42)
                        param_grid = {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "activation": ["relu", "tanh"]}

                    # ======== Grid Search ========
                    st.info("⏳ Running Grid Search... please wait...")
                    grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, return_train_score=True)
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_

                    # ======== Performance ========
                    y_pred = best_model.predict(X_test)
                    y_pred_train = best_model.predict(X_train)

                    acc_train = accuracy_score(y_train, y_pred_train)
                    acc_test = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    st.subheader("🏆 Model Performance Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric": ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1-Score"],
                        "Score": [acc_train, acc_test, prec, rec, f1]
                    })
                    st.table(metrics_df)

                    st.write(f"**Best Parameters:** {grid.best_params_}")

                    # ======== Accuracy Comparison ========
                    st.subheader("📊 Training vs Testing Accuracy")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=["Training", "Testing"], y=[acc_train, acc_test], palette="Blues", ax=ax)
                    ax.set_ylim(0, 1)
                    for i, v in enumerate([acc_train, acc_test]):
                        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
                    st.pyplot(fig)

                    # ======== Parameter Tuning Visual ========
                    st.subheader("🔍 Parameter Tuning Results (Mean CV Accuracy)")
                    grid_df = pd.DataFrame(grid.cv_results_)

                    if len(param_grid.keys()) == 1:
                        param_name = list(param_grid.keys())[0]
                        plt.figure(figsize=(6, 4))
                        plt.plot(grid_df["param_" + param_name], grid_df["mean_test_score"], marker='o')
                        plt.xlabel(param_name)
                        plt.ylabel("Mean CV Accuracy")
                        plt.title(f"{classifier_name} Parameter Tuning")
                        st.pyplot(plt)
                    else:
                        params = list(param_grid.keys())
                        try:
                            pivot_table = grid_df.pivot_table(
                                values="mean_test_score",
                                index="param_" + params[0],
                                columns="param_" + params[1]
                            )
                            plt.figure(figsize=(6, 4))
                            sns.heatmap(pivot_table, annot=True, cmap="Blues")
                            plt.title(f"{classifier_name} Grid Search Accuracy")
                            st.pyplot(plt)
                        except Exception:
                            plt.figure(figsize=(6, 4))
                            plt.plot(grid_df["mean_test_score"], marker='o')
                            plt.title("Mean CV Accuracy per Combination")
                            st.pyplot(plt)

                    # ======== Confusion Matrix ========
                    st.subheader("🔹 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=le.classes_, yticklabels=le.classes_)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(plt)

                    # ======== Feature Importance (All Models) ========
                    st.subheader("🌟 Feature Importance")

                    try:
                        if hasattr(best_model, "feature_importances_"):
                            # Tree-based models (Random Forest)
                            importances = best_model.feature_importances_
                        else:
                            # Model-agnostic feature importance
                            imp_result = permutation_importance(best_model, X_test, y_test, scoring="accuracy", n_repeats=10)
                            importances = imp_result.importances_mean

                        sorted_idx = np.argsort(importances)[::-1]
                        plt.figure(figsize=(8, 4))
                        plt.bar(range(len(importances)), importances[sorted_idx])
                        plt.xticks(range(len(importances)), np.array(selected_features)[sorted_idx], rotation=90)
                        plt.title(f"{classifier_name} Feature Importance")
                        plt.tight_layout()
                        st.pyplot(plt)
                    except Exception as e:
                        st.warning(f"⚠️ Feature importance could not be computed: {e}")

                except Exception as e:
                    st.error(f"❌ Error during classification: {e}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Please upload a CSV or XLSX EEG dataset to start.")
