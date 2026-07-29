import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import asyncio

# ==========================================
# 1. ENVIRONMENT SETUP & DEPENDENCY INSTALLS
# ==========================================
@st.cache_resource
def install_packages():
    try:
        import micropip
        async def install():
            await micropip.install("scikit-learn")
        asyncio.run(install())
    except ImportError:
        pass # Not running in Pyodide/Browser playground

install_packages()

# Now safely import scikit-learn components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

plt.style.use("ggplot")

# ==========================================
# 2. FILE UPLOAD INTERFACE
# ==========================================
st.title("Student Performance Analysis & Modeling")
uploaded_file = st.file_uploader("Upload your student performance CSV file", type=["csv"])

if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file)
    
    # ==========================================
    # 3. DATA EXPLORATION (EDA)
    # ==========================================
    st.header("1. Exploratory Data Analysis")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Dataset Summary Statistics")
    st.write(f"**Shape of Dataset:** {df.shape[0]} rows, {df.shape[1]} columns")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values Per Column:**")
        st.write(df.isnull().sum())
    with col2:
        st.write("**Duplicate Rows:**", df.duplicated().sum())
        st.write("**Data Types:**")
        st.write(df.dtypes.astype(str))
        
    st.dataframe(df.describe())

    # ==========================================
    # 4. VISUALIZATIONS
    # ==========================================
    st.header("2. Data Visualizations")
    
    # Plot 1: Distribution of Final Exam Scores
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["final_exam_score"], bins=20, kde=True, ax=ax)
    plt.title("Distribution of Final Exam Scores")
    st.pyplot(fig)
    plt.close()

    # Plot 2: Gender & Performance Categories
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="gender", ax=ax)
        plt.title("Gender Distribution")
        st.pyplot(fig)
        plt.close()
    with col_chart2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="performance_category", ax=ax)
        plt.title("Performance Categories")
        st.pyplot(fig)
        plt.close()

    # Plot 3: Scatter plots
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="study_hours_per_day", y="final_exam_score", hue="performance_category", ax=ax)
    plt.title("Study Hours vs Final Exam Score")
    st.pyplot(fig)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="attendance_percentage", y="final_exam_score", ax=ax)
    plt.title("Attendance vs Final Exam Score")
    st.pyplot(fig)
    plt.close()

    # Plot 4: Social Media
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["social_media_hours"], bins=20, kde=True, ax=ax)
    plt.title("Distribution of Daily Social Media Usage")
    st.pyplot(fig)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="social_media_hours", y="final_exam_score", hue="performance_category", ax=ax)
    plt.title("Social Media Usage Vs Final Exam Score")
    st.pyplot(fig)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="performance_category", y="social_media_hours", ax=ax)
    plt.title("Social Media Usage Across Performance Categories")
    st.pyplot(fig)
    plt.close()

    # Feature Engineering: Doomscrolling Transformation & Plot
    if "doomscrolling_before_sleep" in df.columns:
        df["doomscrolling_before_sleep"] = df["doomscrolling_before_sleep"].replace({0: "No", 1: "Yes"})
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="doomscrolling_before_sleep", order=["No", "Yes"], color="skyblue", ax=ax)
        plt.title("Students Who Doomscroll Before Sleeping")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
        plt.close()

    # Feature Engineering: AI Dependency
    if 'ai_tool_usage_hours' in df.columns and 'study_hours_per_day' in df.columns:
        df['ai_dependency'] = df['ai_tool_usage_hours'] / (df['study_hours_per_day'] + 1)

    # Boxplot Gender vs Performance Category
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="gender", y="performance_category" if df["performance_category"].dtype != 'object' else "final_exam_score", data=df, ax=ax)
    plt.title("Gender Comparison Analysis")
    st.pyplot(fig)
    plt.close()

    # ==========================================
    # 5. MACHINE LEARNING MODELLING
    # ==========================================
    st.header("3. Machine Learning Models")

    # Encode categorical variables for modeling
    df_encoded = df.copy()
    label = LabelEncoder()
    for col in df_encoded.select_dtypes(include="object").columns:
        df_encoded[col] = label.fit_transform(df_encoded[col])

    # Check for target column
    if "performance_category" in df_encoded.columns:
        x = df_encoded.drop("performance_category", axis=1)
        y = df_encoded["performance_category"]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # --- MODEL 1: Random Forest ---
        st.subheader("Model 1: Random Forest Classifier")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        acc_rf = accuracy_score(y_test, predictions)
        st.metric(label="Random Forest Accuracy", value=f"{acc_rf:.2%}")
        
        st.text("Classification Report:")
        st.text(classification_report(y_test, predictions))

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
        plt.close()

        # Feature Importance
        importance = pd.DataFrame({"Feature": x.columns, "Importance": model.feature_importances_})
        importance = importance.sort_values(by="Importance", ascending=False)
        
        st.write("Top Features:")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=importance.head(10), x="Importance", y="Feature", ax=ax)
        plt.title("Top 10 Most Important Features (RF)")
        st.pyplot(fig)
        plt.close()

        # --- MODEL 2: Decision Tree ---
        st.subheader("Model 2: Decision Tree Classifier")
        model2 = DecisionTreeClassifier(random_state=42)
        model2.fit(X_train, y_train)
        predictions_dt = model2.predict(X_test)
        acc_dt = accuracy_score(y_test, predictions_dt)
        st.metric(label="Decision Tree Accuracy", value=f"{acc_dt:.2%}")

        # --- Preprocessing for Scaled Models (SVC & KNN) ---
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(X_train)
        x_test_scaled = scaler.transform(X_test)

        imputer = SimpleImputer(strategy="mean")
        x_train_scaled = imputer.fit_transform(x_train_scaled)
        x_test_scaled = imputer.transform(x_test_scaled)

        # --- MODEL 3: Support Vector Classifier ---
        st.subheader("Model 3: Support Vector Classifier (SVC)")
        model3 = SVC(random_state=42)
        model3.fit(x_train_scaled, y_train)
        predictions_svc = model3.predict(x_test_scaled)
        acc_svc = accuracy_score(y_test, predictions_svc)
        st.metric(label="SVC Accuracy", value=f"{acc_svc:.2%}")

        # --- MODEL 4: K-Nearest Neighbors ---
        st.subheader("Model 4: K-Nearest Neighbors (KNN)")
        model4 = KNeighborsClassifier()
        model4.fit(x_train_scaled, y_train)
        y_pred_knn = model4.predict(x_test_scaled)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        st.metric(label="KNN Accuracy", value=f"{acc_knn:.2%}")
        
        st.text("KNN Classification Report:")
        st.text(classification_report(y_test, y_pred_knn))
    else:
        st.error("The column 'performance_category' target variable was not found in the file.")

else:
    st.info("👋 Welcome! Please upload your student performance CSV dataset to generate visualizations and train machine learning models.")
