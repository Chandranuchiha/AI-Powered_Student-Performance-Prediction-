import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.inspection import permutation_importance


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Powered Student Performance Prediction")

st.markdown("""
Predict student academic performance using Machine Learning.
Upload your student dataset, train the AI model and predict results.
""")


# =====================================================
# SESSION STATE
# =====================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "target" not in st.session_state:
    st.session_state.target = None

if "features" not in st.session_state:
    st.session_state.features = None

if "task" not in st.session_state:
    st.session_state.task = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "actual" not in st.session_state:
    st.session_state.actual = None

if "predicted" not in st.session_state:
    st.session_state.predicted = None

if "importance" not in st.session_state:
    st.session_state.importance = None


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Upload Dataset",
        "Data Analysis",
        "Train AI Model",
        "Prediction",
        "About Project"
    ]
)


# =====================================================
# DATA PREPROCESSING
# =====================================================

def clean_data(data):

    data = data.copy()

    data.columns = data.columns.astype(str).str.strip()

    data = data.replace(r"^\s*$", np.nan, regex=True)

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data


# =====================================================
# MODEL PIPELINE
# =====================================================

def create_model(X, task):

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numerical_columns
    ]

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numeric",
                numerical_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessing = ColumnTransformer(
        transformers=transformers
    )

    if task == "Regression":

        algorithm = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    else:

        algorithm = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("model", algorithm)
    ])

    return model


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("📊 Student Performance Dashboard")

    if st.session_state.df is None:

        st.info("Please upload a student dataset first.")

    else:

        df = st.session_state.df

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Students",
            len(df)
        )

        col2.metric(
            "Total Columns",
            len(df.columns)
        )

        col3.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )

        col4.metric(
            "Duplicate Records",
            int(df.duplicated().sum())
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        numerical_data = df.select_dtypes(
            include="number"
        )

        if not numerical_data.empty:

            st.subheader("Student Performance Overview")

            selected = st.selectbox(
                "Select numerical column",
                numerical_data.columns
            )

            st.bar_chart(
                numerical_data[selected].dropna()
            )


# =====================================================
# UPLOAD DATASET
# =====================================================

elif page == "Upload Dataset":

    st.header("📁 Upload Student Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        try:

            df = pd.read_csv(uploaded_file)

            if df.empty:

                st.error("Uploaded dataset is empty.")

            elif len(df.columns) < 2:

                st.error(
                    "Dataset must contain at least two columns."
                )

            else:

                df = clean_data(df)

                if df.columns.duplicated().any():

                    st.error(
                        "Duplicate column names found."
                    )

                else:

                    st.session_state.df = df

                    # Reset previously trained model
                    st.session_state.model = None
                    st.session_state.target = None
                    st.session_state.features = None
                    st.session_state.metrics = None
                    st.session_state.actual = None
                    st.session_state.predicted = None
                    st.session_state.importance = None

                    st.success(
                        "Dataset uploaded successfully!"
                    )

                    st.write("Dataset Shape:", df.shape)

                    st.dataframe(
                        df.head(20),
                        use_container_width=True
                    )

        except Exception as error:

            st.error(
                f"Unable to upload dataset: {error}"
            )


# =====================================================
# DATA ANALYSIS
# =====================================================

elif page == "Data Analysis":

    st.header("🔍 Data Analysis & Quality Checks")

    if st.session_state.df is None:

        st.warning("Upload dataset first.")

    else:

        df = st.session_state.df

        st.subheader("Dataset Information")

        st.write("Rows:", len(df))
        st.write("Columns:", len(df.columns))

        st.subheader("Missing Values")

        missing = df.isnull().sum()

        st.dataframe(
            missing.rename("Missing Count")
        )

        if missing.sum() > 0:

            st.bar_chart(missing)

        else:

            st.success("No missing values found.")

        st.subheader("Duplicate Records")

        st.write(
            "Duplicate Rows:",
            int(df.duplicated().sum())
        )

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True
        )

        st.subheader("Correlation")

        numerical = df.select_dtypes(
            include="number"
        )

        if len(numerical.columns) >= 2:

            st.dataframe(
                numerical.corr(),
                use_container_width=True
            )


# =====================================================
# TRAIN AI MODEL
# =====================================================

elif page == "Train AI Model":

    st.header("🤖 Train Machine Learning Model")

    if st.session_state.df is None:

        st.warning("Please upload dataset first.")

    else:

        df = st.session_state.df

        target = st.selectbox(
            "Select Target Column",
            df.columns
        )

        features = st.multiselect(
            "Select Input Features",
            [
                col for col in df.columns
                if col != target
            ],
            default=[
                col for col in df.columns
                if col != target
            ]
        )

        test_size = st.slider(
            "Testing Data Percentage",
            10,
            40,
            20,
            5
        )

        if target:

            if pd.api.types.is_numeric_dtype(df[target]):

                task = "Regression"

            else:

                task = "Classification"

            st.info(
                f"Detected Model Type: {task}"
            )

        if st.button(
            "🚀 Train AI Model",
            type="primary"
        ):

            try:

                if not features:

                    st.error(
                        "Select at least one input feature."
                    )

                else:

                    data = clean_data(
                        df[features + [target]]
                    )

                    data = data.dropna(
                        subset=[target]
                    )

                    if len(data) < 5:

                        st.error(
                            "At least 5 valid records are required."
                        )

                    elif data[target].nunique() < 2:

                        st.error(
                            "Target must contain at least two different values."
                        )

                    else:

                        X = data[features]
                        y = data[target]

                        if pd.api.types.is_numeric_dtype(y):

                            task = "Regression"

                        else:

                            task = "Classification"

                        stratify = None

                        if task == "Classification":

                            counts = y.value_counts()

                            if (
                                counts.min() >= 2
                                and len(y) * test_size / 100
                                >= y.nunique()
                            ):

                                stratify = y

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=test_size / 100,
                            random_state=42,
                            stratify=stratify
                        )

                        if len(X_test) < 1:

                            st.error(
                                "Not enough test records."
                            )

                        else:

                            model = create_model(
                                X_train,
                                task
                            )

                            with st.spinner(
                                "Training AI model..."
                            ):

                                model.fit(
                                    X_train,
                                    y_train
                                )

                                predictions = model.predict(
                                    X_test
                                )

                            # Evaluation

                            if task == "Regression":

                                metrics = {

                                    "MAE": mean_absolute_error(
                                        y_test,
                                        predictions
                                    ),

                                    "RMSE": np.sqrt(
                                        mean_squared_error(
                                            y_test,
                                            predictions
                                        )
                                    ),

                                    "R2 Score": (
                                        r2_score(
                                            y_test,
                                            predictions
                                        )
                                        if len(y_test) > 1
                                        else np.nan
                                    )
                                }

                            else:

                                metrics = {

                                    "Accuracy": accuracy_score(
                                        y_test,
                                        predictions
                                    ),

                                    "Precision": precision_score(
                                        y_test,
                                        predictions,
                                        average="weighted",
                                        zero_division=0
                                    ),

                                    "Recall": recall_score(
                                        y_test,
                                        predictions,
                                        average="weighted",
                                        zero_division=0
                                    ),

                                    "F1 Score": f1_score(
                                        y_test,
                                        predictions,
                                        average="weighted",
                                        zero_division=0
                                    )
                                }

                            st.session_state.model = model
                            st.session_state.target = target
                            st.session_state.features = features
                            st.session_state.task = task
                            st.session_state.metrics = metrics
                            st.session_state.actual = y_test.reset_index(
                                drop=True
                            )
                            st.session_state.predicted = predictions

                            st.session_state.importance = None

                            # Feature importance
                            try:

                                scoring = (
                                    "r2"
                                    if task == "Regression"
                                    else "accuracy"
                                )

                                importance = permutation_importance(
                                    model,
                                    X_test,
                                    y_test,
                                    n_repeats=5,
                                    random_state=42,
                                    scoring=scoring,
                                    n_jobs=-1
                                )

                                st.session_state.importance = (
                                    pd.DataFrame({
                                        "Feature": features,
                                        "Importance": importance.importances_mean
                                    })
                                    .sort_values(
                                        "Importance",
                                        ascending=False
                                    )
                                )

                            except Exception:

                                st.session_state.importance = None

                            st.success(
                                "AI Model Trained Successfully!"
                            )

            except Exception as error:

                st.error(
                    f"Model training failed: {error}"
                )

        # Evaluation results

        if st.session_state.model is not None:

            st.divider()

            st.subheader("📈 Model Evaluation")

            metrics = st.session_state.metrics

            columns = st.columns(len(metrics))

            for column, (name, value) in zip(
                columns,
                metrics.items()
            ):

                display_value = (
                    f"{value:.4f}"
                    if pd.notna(value)
                    else "Unavailable"
                )

                column.metric(
                    name,
                    display_value
                )

            actual = st.session_state.actual

            predicted = st.session_state.predicted

            comparison = pd.DataFrame({

                "Actual": actual.astype(str),

                "Predicted": pd.Series(
                    predicted
                ).astype(str)

            })

            st.subheader("Actual vs Predicted Results")

            st.dataframe(
                comparison,
                use_container_width=True
            )

            if st.session_state.task == "Regression":

                chart_data = pd.DataFrame({

                    "Actual": actual,

                    "Predicted": predicted

                })

                st.subheader("Prediction Comparison")

                st.scatter_chart(
                    chart_data,
                    x="Actual",
                    y="Predicted"
                )

            else:

                labels = sorted(
                    actual.astype(str).unique()
                )

                cm = confusion_matrix(
                    actual.astype(str),
                    pd.Series(predicted).astype(str),
                    labels=labels
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=labels,
                    columns=labels
                )

                st.subheader("Confusion Matrix")

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

                with st.expander(
                    "Classification Report"
                ):

                    st.text(
                        classification_report(
                            actual,
                            predicted,
                            zero_division=0
                        )
                    )

            if st.session_state.importance is not None:

                st.subheader("Feature Importance")

                st.caption(
                    "This shows how much the model relies on each feature. "
                    "It does not prove causation."
                )

                st.dataframe(
                    st.session_state.importance,
                    use_container_width=True
                )

                st.bar_chart(
                    st.session_state.importance.set_index(
                        "Feature"
                    )
                )


# =====================================================
# PREDICTION
# =====================================================

elif page == "Prediction":

    st.header("🔮 Student Performance Prediction")

    if st.session_state.model is None:

        st.warning(
            "Please train the AI model first."
        )

    else:

        model = st.session_state.model

        features = st.session_state.features

        st.write(
            "Upload a new CSV file containing the same input features."
        )

        uploaded = st.file_uploader(
            "Upload Student Data for Prediction",
            type=["csv"],
            key="prediction_file"
        )

        if uploaded:

            try:

                new_data = pd.read_csv(uploaded)

                missing_columns = [
                    col for col in features
                    if col not in new_data.columns
                ]

                if missing_columns:

                    st.error(
                        "Missing columns: "
                        + ", ".join(missing_columns)
                    )

                elif new_data.empty:

                    st.error(
                        "Uploaded file is empty."
                    )

                else:

                    input_data = clean_data(
                        new_data[features]
                    )

                    predictions = model.predict(
                        input_data
                    )

                    output = new_data.copy()

                    output[
                        "Predicted_" + st.session_state.target
                    ] = predictions

                    st.success(
                        "Predictions generated successfully!"
                    )

                    st.subheader(
                        "Student Prediction Results"
                    )

                    st.dataframe(
                        output,
                        use_container_width=True
                    )

                    csv = output.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(

                        label="⬇️ Download Prediction Results",

                        data=csv,

                        file_name="student_predictions.csv",

                        mime="text/csv"

                    )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.header("📚 About the Project")

    st.markdown("""
    ### AI-Powered Student Performance Prediction

    **Objective**

    Analyse historical academic records to estimate student
    performance and identify patterns that may help educators
    plan timely academic support.

    ### Technologies Used

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Random Forest

    ### Machine Learning Algorithms

    1. Random Forest Regression
    2. Random Forest Classification

    ### Data Preprocessing

    - Missing value handling
    - Numerical data scaling
    - Categorical encoding
    - Data validation

    ### Evaluation Metrics

    **Regression:**
    - Mean Absolute Error
    - Root Mean Squared Error
    - R2 Score

    **Classification:**
    - Accuracy
    - Precision
    - Recall
    - F1 Score

    ### Limitations

    - Results depend on the quality and size of the uploaded dataset.
    - Predictions are estimates, not guaranteed academic outcomes.
    - Feature importance does not establish cause and effect.
    - Use only synthetic, anonymised, or authorised student records.
    """)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Powered Student Performance Prediction | "
    "Built using Python, Streamlit and Machine Learning"
)
