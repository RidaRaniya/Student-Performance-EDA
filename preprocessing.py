# ==========================================
# STUDENT PERFORMANCE
# DATA PREPROCESSING
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/student_data.csv")

print("\n========== ORIGINAL DATA ==========")
print(df.head())


# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())


# ==========================================
# 3. REMOVE DUPLICATE ROWS
# ==========================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

# Numerical columns
numerical_columns = [
    "age",
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "final_score"
]

# Fill missing numerical values with median
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())


# Categorical column
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])


print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())


# ==========================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("final_score", axis=1)

y = df["final_score"]


print("\n========== FEATURES (X) ==========")
print(X.head())

print("\n========== TARGET (y) ==========")
print(y.head())


# ==========================================
# 6. ENCODE CATEGORICAL DATA
# ==========================================

# Convert gender into numerical values
X = pd.get_dummies(
    X,
    columns=["gender"],
    drop_first=True,
    dtype=int
)

print("\n========== ENCODED FEATURES ==========")
print(X.head())

print("\nFeature columns:")
print(X.columns)


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 8. DISPLAY SPLIT INFORMATION
# ==========================================

print("\n========== TRAINING DATA ==========")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


print("\n========== TESTING DATA ==========")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# 9. SAVE PROCESSED DATA
# ==========================================

X_train.to_csv(
    "data/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/y_test.csv",
    index=False
)


# ==========================================
# 10. FINAL MESSAGE
# ==========================================

print("\n========== PREPROCESSING COMPLETED ==========")

print("Training and testing datasets have been saved.")
