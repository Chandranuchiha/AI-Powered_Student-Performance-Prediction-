import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------
# AI-Powered Student Performance Prediction
# Streamlit prototype based on the supplied project brief.
# Upload a CSV, select the academic outcome to predict,
# train a baseline model, inspect results, and export predictions.
# ---------------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Powered Student Performance Prediction")
st.caption(
    "A baseline machine-learning prototype for analysing academic records "
    "and estimating a selected academic outcome."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Dataset & Quality", "Train & Evaluate", "Predict & Export", "About"]
    )
    st.divider()
    st.caption("Use synthetic, anonymised, or explicitly authorised student data only.")

# Keep the uploaded data and trained pipeline during the Streamlit session.
if "student_df" not in st.session_state:
    st.session_state.student_df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "target" not in st.session_state:
    st.session_state.target = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None
if "task_type" not in st.session_state:
    st.session_state.task_type = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "test_metrics" not in st.session_state:
    st.session_state.test_metrics = None
if "test_predictions" not in st.session_state:
    st.session_state.test_predictions = None
if "test_actual" not in st.session_state:
    st.session_state.test_actual = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None
if "prediction_frame" not in st.session_state:
    st.session_state.prediction_frame = None


def show_dataset_loader():
    st.subheader("Upload student dataset")
    st.write("Upload a CSV file containing student academic records.")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="The first row should contain column names."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded CSV is empty. Please choose a file with records.")
                return
            if len(df.columns) < 2:
                st.error("The CSV must contain at least two columns.")
                return

            # Remove accidental whitespace around column names.
            df.columns = [str(c).strip() for c in df.columns]
            if df.columns.duplicated().any():
                st.error("The file contains duplicate column names. Rename them and upload again.")
                return

            st.session_state.student_df = df
            # A new dataset invalidates any model trained on the previous dataset.
            st.session_state.model = None
            st.session_state.target = None
            st.session_state.feature_columns = None
            st.session_state.test_metrics = None
            st.session_state.test_predictions = None
            st.session_state.test_actual = None
            st.session_state.feature_importance = None
            st.session_state.prediction_frame = None
            st.success(f"Dataset loaded: {len(df):,} rows and {len(df.columns)} columns.")
        except Exception as exc:
            st.error(f"Could not read this CSV file: {exc}")


def get_task_type(series):
    """Numeric outcomes use regression; other outcomes use classification."""
    return "regression" if pd.api.types.is_numeric_dtype(series) else "classification"


def build_pipeline(X, task_type):
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers)

    if task_type == "regression":
        estimator = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])


def clean_features(frame):
    """Normalise empty strings and infinite numeric values before modelling."""
    frame = frame.copy()
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


df = st.session_state.student_df

if page == "Dashboard":
    st.header("Dashboard")
    if df is None:
        st.info("Upload a student CSV dataset to begin.")
        show_dataset_loader()
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Student records", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

        st.subheader("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)

        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            st.subheader("Numeric column distributions")
            selected_column = st.selectbox("Choose a numeric column", numeric_df.columns)
            fig, ax = plt.subplots()
            numeric_df[selected_column].dropna().hist(ax=ax, bins=20)
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Number of records")
            ax.set_title(f"Distribution of {selected_column}")
            st.pyplot(fig)
            plt.close(fig)

elif page == "Dataset & Quality":
    st.header("Dataset & Data Quality")
    if df is None:
        show_dataset_loader()
    else:
        st.success(f"Current dataset: {len(df):,} rows × {len(df.columns)} columns")
        st.subheader("Preview")
        st.dataframe(df.head(30), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Column data types")
            type_table = pd.DataFrame({
                "Column": df.columns,
                "Data type": [str(t) for t in df.dtypes],
                "Missing values": df.isna().sum().values,
                "Missing (%)": (df.isna().mean().values * 100).round(2)
            })
            st.dataframe(type_table, use_container_width=True, hide_index=True)
        with right:
            st.subheader("Quality checks")
            st.metric("Duplicate rows", int(df.duplicated().sum()))
            st.metric("Rows with at least one missing value", int(df.isna().any(axis=1).sum()))
            st.metric("Total missing cells", int(df.isna().sum().sum()))

        st.subheader("Missing values by column")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.bar_chart(missing)

        st.warning(
            "The model pipeline imputes missing feature values during training. "
            "Rows without a target value are excluded from model training. "
            "Review duplicates, ranges, and unusual values with your dataset documentation."
        )

elif page == "Train & Evaluate":
    st.header("Train & Evaluate Model")
    if df is None:
        st.info("Upload a CSV dataset from the Dashboard or Dataset & Quality page.")
    else:
        st.write("Choose the academic outcome column that the model should predict.")
        target = st.selectbox("Target / outcome column", options=list(df.columns))
        available_features = [c for c in df.columns if c != target]
        default_features = available_features
        features = st.multiselect(
            "Input features",
            options=available_features,
            default=default_features,
            help="Do not select student names, IDs, or other unnecessary personal identifiers."
        )
        test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
        st.caption("The split and Random Forest model use random_state=42 for reproducibility.")

        if target:
            task_type = get_task_type(df[target])
            st.info(
                "Detected task: **Regression** (numeric outcome)"
                if task_type == "regression"
                else "Detected task: **Classification** (categorical outcome)"
            )

        if st.button("Train baseline model", type="primary", disabled=(len(features) == 0)):
            try:
                working = clean_features(df[features + [target]])
                before = len(working)
                working = working.dropna(subset=[target])
                removed = before - len(working)

                if len(working) < 5:
                    st.error("At least 5 rows with a valid target are required.")
                elif working[target].nunique() < 2:
                    st.error("The target column must contain at least two distinct values.")
                else:
                    X = working[features]
                    y = working[target]
                    task_type = get_task_type(y)

                    if task_type == "classification":
                        # Stratification is used only when every class has enough records.
                        counts = y.value_counts()
                        stratify_y = y if counts.min() >= 2 else None
                        if stratify_y is not None and len(y) * (test_size / 100) < y.nunique():
                            stratify_y = None
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size / 100,
                            random_state=42, stratify=stratify_y
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size / 100, random_state=42
                        )

                    if len(X_train) < 2 or len(X_test) < 1:
                        st.error("Not enough records for this train/test split.")
                    else:
                        model = build_pipeline(X_train, task_type)
                        with st.spinner("Training and evaluating the model..."):
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)

                        if task_type == "regression":
                            metrics = {
                                "MAE": mean_absolute_error(y_test, predictions),
                                "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                                "R²": r2_score(y_test, predictions) if len(y_test) > 1 else np.nan
                            }
                        else:
                            metrics = {
                                "Accuracy": accuracy_score(y_test, predictions),
                                "Precision (weighted)": precision_score(
                                    y_test, predictions, average="weighted", zero_division=0
                                ),
                                "Recall (weighted)": recall_score(
                                    y_test, predictions, average="weighted", zero_division=0
                                ),
                                "F1 (weighted)": f1_score(
                                    y_test, predictions, average="weighted", zero_division=0
                                )
                            }

                        # Store fitted model and metadata for the prediction page.
                        st.session_state.model = model
                        st.session_state.target = target
                        st.session_state.feature_columns = features
                        st.session_state.task_type = task_type
                        st.session_state.test_metrics = metrics
                        st.session_state.test_predictions = predictions
                        st.session_state.test_actual = y_test.reset_index(drop=True)
                        st.session_state.prediction_frame = None

                        # Permutation importance is model-agnostic and uses held-out records.
                        try:
                            scoring = "r2" if task_type == "regression" else "accuracy"
                            importance = permutation_importance(
                                model, X_test, y_test, n_repeats=5,
                                random_state=42, scoring=scoring, n_jobs=-1
                            )
                            st.session_state.feature_importance = pd.DataFrame({
                                "Feature": features,
                                "Importance": importance.importances_mean
                            }).sort_values("Importance", ascending=False)
                        except Exception:
                            st.session_state.feature_importance = None

                        st.success(
                            f"Model trained successfully. {removed} row(s) with missing target values were excluded."
                        )
            except ValueError as exc:
                st.error(f"Unable to train the model. Check the target values and class counts. Details: {exc}")
            except Exception as exc:
                st.error(f"An unexpected error occurred during training: {exc}")

        if st.session_state.model is not None:
            st.divider()
            st.subheader("Evaluation results on held-out test data")
            metric_columns = st.columns(len(st.session_state.test_metrics))
            for col, (name, value) in zip(metric_columns, st.session_state.test_metrics.items()):
                display_value = f"{value:.4f}" if pd.notna(value) else "Not available"
                col.metric(name, display_value)

            actual = st.session_state.test_actual
            predicted = st.session_state.test_predictions
            comparison = pd.DataFrame({
                "Actual": actual.astype(str),
                "Predicted": pd.Series(predicted).astype(str)
            })
            st.subheader("Actual vs predicted sample")
            st.dataframe(comparison.head(30), use_container_width=True)

            if st.session_state.task_type == "regression":
                fig, ax = plt.subplots()
                ax.scatter(actual, predicted, alpha=0.7)
                low = min(float(np.min(actual)), float(np.min(predicted)))
                high = max(float(np.max(actual)), float(np.max(predicted)))
                ax.plot([low, high], [low, high], linestyle="--")
                ax.set_xlabel("Actual values")
                ax.set_ylabel("Predicted values")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
                plt.close(fig)
            else:
                labels = sorted(pd.Series(actual).astype(str).unique().tolist())
                cm = confusion_matrix(actual.astype(str), pd.Series(predicted).astype(str), labels=labels)
                fig, ax = plt.subplots()
                ax.imshow(cm)
                ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
                ax.set_yticks(range(len(labels)), labels=labels)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                with st.expander("Classification report"):
                    st.text(classification_report(actual, predicted, zero_division=0))

            if st.session_state.feature_importance is not None:
                st.subheader("Factors associated with model predictions")
                st.caption(
                    "Permutation importance measures how model performance changes when a feature is shuffled. "
                    "It indicates model reliance, not causation."
                )
                st.dataframe(st.session_state.feature_importance, use_container_width=True, hide_index=True)
                chart_data = st.session_state.feature_importance.set_index("Feature")
                st.bar_chart(chart_data)

elif page == "Predict & Export":
    st.header("Predict & Export")
    model = st.session_state.model
    if model is None:
        st.info("Train a model first in the Train & Evaluate page.")
    else:
        st.write(f"Model outcome: **{st.session_state.target}**")
        st.write(f"Required input features: {', '.join(st.session_state.feature_columns)}")
        prediction_file = st.file_uploader(
            "Upload a new CSV containing the same input feature columns",
            type=["csv"],
            key="prediction_upload"
        )

        if prediction_file is not None:
            try:
                new_data = pd.read_csv(prediction_file)
                required = st.session_state.feature_columns
                missing_columns = [c for c in required if c not in new_data.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                elif new_data.empty:
                    st.error("The prediction file is empty.")
                else:
                    input_data = clean_features(new_data[required])
                    if input_data.isna().all(axis=1).any():
                        st.warning("Some rows contain no usable feature values. Check the uploaded file.")
                    results = model.predict(input_data)
                    output = new_data.copy()
                    output[f"Predicted_{st.session_state.target}"] = results
                    st.session_state.prediction_frame = output
                    st.success(f"Generated predictions for {len(output):,} records.")
                    st.dataframe(output.head(50), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not generate predictions: {exc}")

        if st.session_state.prediction_frame is not None:
            export_data = st.session_state.prediction_frame.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=export_data,
                file_name="student_performance_predictions.csv",
                mime="text/csv"
            )

elif page == "About":
    st.header("About this application")
    st.markdown("""
    **Objective:** Analyse historical academic records to estimate a selected
    academic outcome and identify patterns that may help educators plan timely support.

    **Baseline approach**
    - Numeric target column: Random Forest Regression.
    - Non-numeric target column: Random Forest Classification.
    - Numeric missing feature values: median imputation.
    - Categorical missing feature values: most-frequent imputation.
    - Categorical inputs: one-hot encoding.
    - Evaluation: held-out test split with regression or classification metrics.

    **Important limitations**
    - The project brief does not specify a fixed dataset schema or a particular
      academic target column. Select the appropriate target and input features
      based on your dataset and mentor-approved definition of the outcome.
    - This is a baseline prototype, not a validated student assessment system.
    - Metrics depend on the uploaded dataset and test split. No accuracy or
      real-world performance is assumed in advance.
    - Feature importance shows association with model behaviour, not causation.
    - Do not upload identifiable student records unless you are authorised to do so.
    """)

st.divider()
st.caption("AI-Powered Student Performance Prediction | Baseline Streamlit prototype")
