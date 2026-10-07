# # ==========================================
# # STUDENT PERFORMANCE
# # MACHINE LEARNING - LINEAR REGRESSION
# # ==========================================

# import pandas as pd

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # ==========================================
# # 1. LOAD PROCESSED DATA
# # ==========================================

# X_train = pd.read_csv("data/X_train.csv")
# X_test = pd.read_csv("data/X_test.csv")

# y_train = pd.read_csv("data/y_train.csv").squeeze()
# y_test = pd.read_csv("data/y_test.csv").squeeze()


# # ==========================================
# # 2. DISPLAY DATA
# # ==========================================

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)

# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)


# # ==========================================
# # 3. CREATE MODEL
# # ==========================================

# model = LinearRegression()


# # ==========================================
# # 4. TRAIN MODEL
# # ==========================================

# model.fit(X_train, y_train)

# print("\nModel training completed!")


# # ==========================================
# # 5. MAKE PREDICTIONS
# # ==========================================

# y_pred = model.predict(X_test)

# print("\nPredicted values:")
# print(y_pred)

# print("\nActual values:")
# print(y_test.values)


# # ==========================================
# # 6. MODEL EVALUATION
# # ==========================================

# mae = mean_absolute_error(y_test, y_pred)

# mse = mean_squared_error(y_test, y_pred)

# rmse = mse ** 0.5

# r2 = r2_score(y_test, y_pred)


# print("\n========== MODEL PERFORMANCE ==========")

# print("MAE :", mae)
# print("MSE :", mse)
# print("RMSE:", rmse)
# print("R²  :", r2)

# ==========================================
# STUDENT PERFORMANCE
# MODEL COMPARISON
# ==========================================

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD PROCESSED DATA
# ==========================================

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()


# ==========================================
# 2. CREATE MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 3. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

for name, model in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    # Print results
    print("MAE :", round(mae, 2))
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 2))

    # Save results
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })


# ==========================================
# 4. CREATE RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df)


# ==========================================
# 5. FIND BEST MODEL
# ==========================================

best_model = results_df.loc[
    results_df["R2"].idxmax()
]

print("\n========== BEST MODEL ==========")

print(
    "Best Model:",
    best_model["Model"]
)

print(
    "R²:",
    round(best_model["R2"], 2)
)

print(
    "RMSE:",
    round(best_model["RMSE"], 2)
)
