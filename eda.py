import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/student_data.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

plt.figure(figsize=(8, 5))

sns.histplot(
    df["final_score"],
    bins=10,
    kde=True
)

plt.title("Distribution of Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")

plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    df["study_hours"],
    bins=8,
    kde=True
)

plt.title("Distribution of Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")

plt.show()

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=df["study_hours"]
)

plt.title("Study Hours - Box Plot")
plt.xlabel("Study Hours")

plt.show()

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="study_hours",
    y="final_score"
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.show()

plt.figure(figsize=(8, 5))

sns.regplot(
    data=df,
    x="study_hours",
    y="final_score"
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.show()

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="attendance",
    y="final_score"
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")

plt.show()

plt.figure(figsize=(8, 5))

sns.regplot(
    data=df,
    x="attendance",
    y="final_score"
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")

plt.show()

correlation = df.select_dtypes(
    include=np.number
).corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

print("\n========== GENDER DISTRIBUTION ==========")

print(
    df["gender"].value_counts()
)

plt.figure(figsize=(6, 5))

sns.countplot(
    data=df,
    x="gender"
)

plt.title("Number of Students by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Students")

plt.show()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="gender",
    y="final_score"
)

plt.title("Final Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Final Score")

plt.show()

print("\n========== AVERAGE FINAL SCORE BY GENDER ==========")

print(
    df.groupby("gender")["final_score"].mean()
)

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=df["study_hours"]
)

plt.title("Outliers in Study Hours")
plt.xlabel("Study Hours")

plt.show()

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=df["attendance"]
)

plt.title("Outliers in Attendance")
plt.xlabel("Attendance (%)")

plt.show()

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=df["final_score"]
)

plt.title("Outliers in Final Score")
plt.xlabel("Final Score")

plt.show()

print("\n========== EDA SUMMARY ==========")

print(
    "Number of students:",
    df.shape[0]
)

print(
    "Number of variables:",
    df.shape[1]
)

print(
    "Total missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Average final score:",
    round(df["final_score"].mean(), 2)
)

print(
    "Average study hours:",
    round(df["study_hours"].mean(), 2)
)

print(
    "Average attendance:",
    round(df["attendance"].mean(), 2)
)

print("\n========== EDA COMPLETED ==========")
