# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


plt.style.use("ggplot")


# Load dataset

df = pd.read_csv("sample_data/student_performance_dataset.csv")

print("Shape:", df.shape)

df.head()



# Basic information

df.info()

print(df.describe())

print(df.isnull().sum())

print("Duplicates:", df.duplicated().sum())



# Feature Engineering

df["ai_dependency"] = (
    df["ai_tool_usage_hours"] /
    (df["study_hours_per_day"] + 1)
)



# Convert binary column

df["doomscrolling_before_sleep"] = df[
    "doomscrolling_before_sleep"
].replace({
    0:"No",
    1:"Yes"
})



# Visualizations


plt.figure(figsize=(8,5))
sns.histplot(
    df["final_exam_score"],
    bins=20,
    kde=True
)
plt.title("Distribution of Final Exam Scores")
plt.show()



plt.figure(figsize=(6,4))
sns.countplot(
    data=df,
    x="performance_category"
)

plt.title("Performance Categories")
plt.show()



plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="study_hours_per_day",
    y="final_exam_score",
    hue="performance_category"
)

plt.title("Study Hours vs Exam Score")
plt.show()



# Encode categorical variables

label_encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col])



# Split features and target

X = df.drop(
    "performance_category",
    axis=1
)

y = df["performance_category"]



# Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Handle missing values

imputer = SimpleImputer(
    strategy="mean"
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)



# ==========================
# Random Forest
# ==========================

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)


print("\nRandom Forest Accuracy:")
print(
    accuracy_score(
        y_test,
        rf_predictions
    )
)


print(
    classification_report(
        y_test,
        rf_predictions
    )
)



# Confusion Matrix

cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()



# Feature Importance

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        rf_model.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print(
    importance.head(10)
)



plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Important Features"
)

plt.show()



# ==========================
# Decision Tree
# ==========================

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)


dt_predictions = dt_model.predict(
    X_test
)


print(
    "Decision Tree Accuracy:",
    accuracy_score(
        y_test,
        dt_predictions
    )
)



# ==========================
# Scaling for SVM/KNN
# ==========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)



# ==========================
# SVM
# ==========================

svm_model = SVC(
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)


svm_predictions = svm_model.predict(
    X_test_scaled
)


print(
    "SVM Accuracy:",
    accuracy_score(
        y_test,
        svm_predictions
    )
)



# ==========================
# KNN
# ==========================

knn_model = KNeighborsClassifier()

knn_model.fit(
    X_train_scaled,
    y_train
)


knn_predictions = knn_model.predict(
    X_test_scaled
)


print(
    "KNN Accuracy:",
    accuracy_score(
        y_test,
        knn_predictions
    )
)



print(
    classification_report(
        y_test,
        knn_predictions
    )
)