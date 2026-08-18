import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "plot"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "report"
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

FEATURES = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

TARGET = "Salary Package"


# ============================================================
# LOAD DATA
# ============================================================

def load_salary_data():

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Original dataset shape:", df.shape)

    required_columns = FEATURES + [TARGET]

    data = df[required_columns].copy()

    # Convert target to numeric
    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce"
    )

    # Remove missing values
    data = data.dropna(
        subset=required_columns
    )

    # Salary should be greater than 0
    data = data[
        data[TARGET] > 0
    ].copy()

    print(
        "Rows available for salary prediction:",
        len(data)
    )

    return data


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize(X):

    mean = X.mean(axis=0)

    std = X.std(
        axis=0,
        ddof=0
    )

    # Prevent division by zero
    std = np.where(
        std == 0,
        1.0,
        std
    )

    X_scaled = (
        X - mean
    ) / std

    return X_scaled, mean, std


# ============================================================
# BATCH GRADIENT DESCENT
# ============================================================

def batch_gradient_descent(
        X,
        y,
        alpha=0.1,
        epochs=2000
):

    m, n = X.shape

    # Initialize theta
    theta = np.zeros(
        n + 1
    )

    # Add bias column
    X_bias = np.column_stack(
        (
            np.ones(m),
            X
        )
    )

    mse_history = []

    for epoch in range(epochs):

        # Prediction
        y_hat = X_bias @ theta

        # Error
        error = y_hat - y

        # Cost
        mse = np.mean(
            error ** 2
        )

        mse_history.append(
            mse
        )

        # Gradient
        gradient = (
            (2 / m)
            *
            (X_bias.T @ error)
        )

        # Update theta
        theta = (
            theta
            -
            alpha * gradient
        )

    return theta, mse_history


# ============================================================
# CONVERT STANDARDIZED COEFFICIENTS
# ============================================================

def unstandardize_parameters(
        theta,
        mean,
        std
):

    coefficients = (
        theta[1:] / std
    )

    intercept = (
        theta[0]
        -
        np.sum(
            coefficients * mean
        )
    )

    return intercept, coefficients


# ============================================================
# NORMAL EQUATION
# ============================================================

def normal_equation(X, y):

    # Add bias column
    X_bias = np.column_stack(
        (
            np.ones(len(X)),
            X
        )
    )

    # Normal Equation:
    #
    # theta = (X^T X)^-1 X^T y
    #
    # pinv is used for numerical stability.

    theta = (
        np.linalg.pinv(
            X_bias.T @ X_bias
        )
        @
        X_bias.T
        @
        y
    )

    intercept = theta[0]

    coefficients = theta[1:]

    return (
        intercept,
        coefficients
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
        y_true,
        y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mse
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }


# ============================================================
# SAVE PLOTS
# ============================================================

def save_plots(
        y_test,
        y_pred
):

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Actual vs Predicted
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Salary Package (LPA)"
    )

    plt.ylabel(
        "Predicted Salary Package (LPA)"
    )

    plt.title(
        "Actual vs Predicted Salary Package"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "linear_regression_actual_vs_predicted.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Residual Histogram
    # --------------------------------------------------------

    residuals = (
        y_test
        -
        y_pred
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        residuals,
        bins=30
    )

    plt.xlabel(
        "Residual (Actual - Predicted)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Linear Regression Residual Histogram"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "linear_regression_residual_histogram.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN LINEAR REGRESSION FUNCTION
# ============================================================

def run_linear_regression():

    # Load data
    data = load_salary_data()

    # Features
    X = data[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    # Target
    y = data[
        TARGET
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ========================================================
    # BATCH GRADIENT DESCENT
    # ========================================================

    X_train_scaled, mean, std = standardize(
        X_train
    )

    X_test_scaled = (
        X_test - mean
    ) / std

    theta, mse_history = batch_gradient_descent(
        X_train_scaled,
        y_train,
        alpha=0.1,
        epochs=2000
    )

    # Convert coefficients back to original scale
    gd_intercept, gd_coefficients = (
        unstandardize_parameters(
            theta,
            mean,
            std
        )
    )

    # Prediction
    gd_predictions = (
        gd_intercept
        +
        X_test @ gd_coefficients
    )

    # Metrics
    gd_metrics = calculate_metrics(
        y_test,
        gd_predictions
    )

    # ========================================================
    # NORMAL EQUATION
    # ========================================================

    ne_intercept, ne_coefficients = (
        normal_equation(
            X_train,
            y_train
        )
    )

    ne_predictions = (
        ne_intercept
        +
        X_test @ ne_coefficients
    )

    ne_metrics = calculate_metrics(
        y_test,
        ne_predictions
    )

    # ========================================================
    # SCIKIT-LEARN
    # ========================================================

    sklearn_model = LinearRegression()

    sklearn_model.fit(
        X_train,
        y_train
    )

    sklearn_predictions = (
        sklearn_model.predict(
            X_test
        )
    )

    sklearn_metrics = calculate_metrics(
        y_test,
        sklearn_predictions
    )

    # ========================================================
    # SAVE PLOTS
    # ========================================================

    save_plots(
        pd.Series(y_test),
        pd.Series(sklearn_predictions)
    )

    # ========================================================
    # SAVE REPORTS
    # ========================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    # Gradient Descent MSE history
    history_df = pd.DataFrame(
        {
            "Epoch": range(
                1,
                len(mse_history) + 1
            ),
            "MSE": mse_history
        }
    )

    history_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "gradient_descent_mse_history.csv"
        ),
        index=False
    )

    # Model comparison
    comparison_df = pd.DataFrame(
        [
            gd_metrics,
            ne_metrics,
            sklearn_metrics
        ],
        index=[
            "Batch Gradient Descent",
            "Normal Equation",
            "Scikit-learn LinearRegression"
        ]
    )

    comparison_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "linear_regression_model_comparison.csv"
        )
    )

    # Coefficients
    coefficients_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Gradient Descent": gd_coefficients,
            "Normal Equation": ne_coefficients,
            "Scikit-learn": sklearn_model.coef_
        }
    )

    coefficients_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "linear_regression_coefficients.csv"
        ),
        index=False
    )

    # ========================================================
    # NEW STUDENT PREDICTION
    # ========================================================

    new_student = np.array(
        [
            [
                8.5,
                80,
                85,
                75
            ]
        ]
    )

    predicted_salary = (
        sklearn_model.predict(
            new_student
        )[0]
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "rows_used":
            int(len(data)),

        "train_rows":
            int(len(X_train)),

        "test_rows":
            int(len(X_test)),

        "features":
            FEATURES,

        "target":
            TARGET,

        "gd_intercept":
            float(gd_intercept),

        "gd_coefficients":
            gd_coefficients.tolist(),

        "gd_metrics":
            gd_metrics,

        "normal_equation_intercept":
            float(ne_intercept),

        "normal_equation_coefficients":
            ne_coefficients.tolist(),

        "normal_equation_metrics":
            ne_metrics,

        "sklearn_intercept":
            float(
                sklearn_model.intercept_
            ),

        "sklearn_coefficients":
            sklearn_model.coef_.tolist(),

        "sklearn_metrics":
            sklearn_metrics,

        "new_student":
            new_student[0].tolist(),

        "new_prediction":
            float(
                predicted_salary
            ),

        "plot_actual_predicted":
            "/plots/linear_regression_actual_vs_predicted.png",

        "plot_residuals":
            "/plots/linear_regression_residual_histogram.png"
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    result = run_linear_regression()

    print("\n======================================")
    print("PLACEMENT PREDICTION")
    print("LINEAR REGRESSION")
    print("======================================")

    print(
        "\nRows used:",
        result["rows_used"]
    )

    print(
        "Training rows:",
        result["train_rows"]
    )

    print(
        "Testing rows:",
        result["test_rows"]
    )

    print("\nFeatures:")

    for feature in FEATURES:
        print(
            "-",
            feature
        )

    print("\n======================================")
    print("MODEL PERFORMANCE")
    print("======================================")

    models = [
        (
            "Batch Gradient Descent",
            result["gd_metrics"]
        ),
        (
            "Normal Equation",
            result["normal_equation_metrics"]
        ),
        (
            "Scikit-learn",
            result["sklearn_metrics"]
        )
    ]

    for model_name, values in models:

        print(
            "\n",
            model_name
        )

        print(
            "MSE :",
            round(values["MSE"], 4)
        )

        print(
            "RMSE:",
            round(values["RMSE"], 4)
        )

        print(
            "MAE :",
            round(values["MAE"], 4)
        )

        print(
            "R2  :",
            round(values["R2"], 4)
        )

    print("\n======================================")
    print("SCIKIT-LEARN EQUATION")
    print("======================================")

    equation = (
        f"Salary = "
        f"{result['sklearn_intercept']:.4f}"
    )

    for feature, coefficient in zip(
        FEATURES,
        result["sklearn_coefficients"]
    ):

        sign = (
            "+"
            if coefficient >= 0
            else "-"
        )

        equation += (
            f" {sign} "
            f"{abs(coefficient):.4f}"
            f"*{feature}"
        )

    print(equation)

    print("\n======================================")
    print("NEW STUDENT PREDICTION")
    print("======================================")

    print(
        "CGPA:",
        result["new_student"][0]
    )

    print(
        "Aptitude:",
        result["new_student"][1]
    )

    print(
        "Coding:",
        result["new_student"][2]
    )

    print(
        "Interview:",
        result["new_student"][3]
    )

    print(
        "\nPredicted Salary:",
        round(
            result["new_prediction"],
            2
        ),
        "LPA"
    )