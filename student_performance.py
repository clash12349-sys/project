import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.style.use("ggplot")

# ==========================================
# STREAMLIT PAGE
# ==========================================
st.set_page_config(page_title="Student Performance Analysis", layout="wide")

st.title("📊 Student Performance Analysis Dashboard")
st.write("Upload your student performance dataset to perform Exploratory Data Analysis (EDA).")

uploaded_file = st.file_uploader(
    "Upload your student performance CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # ===============================
    # LOAD DATA
    # ===============================
    df = pd.read_csv(uploaded_file)

    # ===============================
    # DATA PREVIEW
    # ===============================
    st.header("1. Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicates", df.duplicated().sum())

    st.subheader("Data Types")
    st.write(df.dtypes.astype(str))

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all"))

    # ===============================
    # CORRELATION
    # ===============================
    st.header("2. Correlation Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close()

    # ===============================
    # FINAL EXAM SCORE
    # ===============================
    if "final_exam_score" in df.columns:

        st.header("3. Final Exam Score")

        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(
            df["final_exam_score"],
            bins=20,
            kde=True,
            ax=ax
        )
        plt.title("Distribution of Final Exam Scores")
        st.pyplot(fig)
        plt.close()

    # ===============================
    # GENDER
    # ===============================
    if "gender" in df.columns:

        st.header("4. Gender Distribution")

        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            data=df,
            x="gender",
            ax=ax
        )
        plt.title("Gender Distribution")
        st.pyplot(fig)
        plt.close()

    # ===============================
    # PERFORMANCE CATEGORY
    # ===============================
    if "performance_category" in df.columns:

        st.header("5. Performance Categories")

        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            data=df,
            x="performance_category",
            ax=ax
        )
        plt.title("Performance Category")
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close()

    # ===============================
    # STUDY HOURS
    # ===============================
    if (
        "study_hours_per_day" in df.columns
        and "final_exam_score" in df.columns
    ):

        st.header("6. Study Hours vs Final Score")

        fig, ax = plt.subplots(figsize=(8,5))

        if "performance_category" in df.columns:
            sns.scatterplot(
                data=df,
                x="study_hours_per_day",
                y="final_exam_score",
                hue="performance_category",
                ax=ax
            )
        else:
            sns.scatterplot(
                data=df,
                x="study_hours_per_day",
                y="final_exam_score",
                ax=ax
            )

        plt.title("Study Hours vs Final Exam Score")
        st.pyplot(fig)
        plt.close()

    # ===============================
    # ATTENDANCE
    # ===============================
    if (
        "attendance_percentage" in df.columns
        and "final_exam_score" in df.columns
    ):

        st.header("7. Attendance vs Final Score")

        fig, ax = plt.subplots(figsize=(8,5))

        sns.scatterplot(
            data=df,
            x="attendance_percentage",
            y="final_exam_score",
            ax=ax
        )

        plt.title("Attendance vs Final Exam Score")
        st.pyplot(fig)
        plt.close()

    # ===============================
    # SOCIAL MEDIA
    # ===============================
    if "social_media_hours" in df.columns:

        st.header("8. Social Media Analysis")

        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(
            df["social_media_hours"],
            bins=20,
            kde=True,
            ax=ax
        )
        plt.title("Daily Social Media Usage")
        st.pyplot(fig)
        plt.close()

        if "final_exam_score" in df.columns:

            fig, ax = plt.subplots(figsize=(8,5))

            if "performance_category" in df.columns:
                sns.scatterplot(
                    data=df,
                    x="social_media_hours",
                    y="final_exam_score",
                    hue="performance_category",
                    ax=ax
                )
            else:
                sns.scatterplot(
                    data=df,
                    x="social_media_hours",
                    y="final_exam_score",
                    ax=ax
                )

            plt.title("Social Media vs Final Exam Score")
            st.pyplot(fig)
            plt.close()

        if "performance_category" in df.columns:

            fig, ax = plt.subplots(figsize=(8,5))

            sns.boxplot(
                data=df,
                x="performance_category",
                y="social_media_hours",
                ax=ax
            )

            plt.title("Social Media Across Performance Categories")
            st.pyplot(fig)
            plt.close()

    # ===============================
    # DOOMSCROLLING
    # ===============================
    if "doomscrolling_before_sleep" in df.columns:

        st.header("9. Doomscrolling")

        temp = df.copy()

        temp["doomscrolling_before_sleep"] = temp[
            "doomscrolling_before_sleep"
        ].replace({
            0: "No",
            1: "Yes"
        })

        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(
            data=temp,
            x="doomscrolling_before_sleep",
            order=["No","Yes"],
            ax=ax
        )

        plt.title("Students Who Doomscroll Before Sleeping")

        for container in ax.containers:
            ax.bar_label(container)

        st.pyplot(fig)
        plt.close()

    # ===============================
    # AI DEPENDENCY
    # ===============================
    if (
        "ai_tool_usage_hours" in df.columns
        and "study_hours_per_day" in df.columns
    ):

        st.header("10. AI Dependency")

        temp = df.copy()

        temp["ai_dependency"] = (
            temp["ai_tool_usage_hours"]
            /
            (temp["study_hours_per_day"] + 1)
        )

        fig, ax = plt.subplots(figsize=(8,5))

        sns.histplot(
            temp["ai_dependency"],
            bins=20,
            kde=True,
            ax=ax
        )

        plt.title("AI Dependency Index")

        st.pyplot(fig)
        plt.close()

    st.success("✅ EDA completed successfully!")

else:
    st.info("Upload a CSV dataset to begin analysis.")
