from flask import Flask, render_template, send_from_directory
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Output"
)

PLOT_PATH = os.path.join(
    OUTPUT_PATH,
    "plot"
)

REPORT_PATH = os.path.join(
    OUTPUT_PATH,
    "report"
)

MODULE3_PATH = os.path.join(
    BASE_DIR,
    "src",
    "module3"
)

MODULE4_PATH = os.path.join(
    BASE_DIR,
    "src",
    "module4"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# HELPER
# ============================================================

def existing_plot(relative_path, title):

    full_path = os.path.join(
        BASE_DIR,
        relative_path
    )

    if os.path.exists(full_path):

        return {
            "path": "/files/" + relative_path.replace("\\", "/"),
            "title": title
        }

    return None


def collect_plots(items):

    result = []

    for item in items:

        if item is not None:
            result.append(item)

    return result


# ============================================================
# REGRESSION CALCULATION
# ============================================================

def calculate_regression_results():

    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    target = "Salary Package"

    data = df[
        features + [target]
    ].copy()

    data[target] = pd.to_numeric(
        data[target],
        errors="coerce"
    )

    data = data.dropna(
        subset=[target]
    )

    data = data[
        data[target] > 0
    ]

    X = data[features]

    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(alpha=1.0),

        "Lasso Regression":
            Lasso(
                alpha=0.1,
                max_iter=10000
            )
    }

    results = []

    coefficients = []

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        mse = mean_squared_error(
            y_test,
            prediction
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        r2 = r2_score(
            y_test,
            prediction
        )

        results.append({

            "model": name,

            "mse": round(
                mse,
                4
            ),

            "rmse": round(
                rmse,
                4
            ),

            "mae": round(
                mae,
                4
            ),

            "r2": round(
                r2,
                4
            )
        })

        for feature, coefficient in zip(
            features,
            model.coef_
        ):

            coefficients.append({

                "model": name,

                "feature": feature,

                "coefficient": round(
                    coefficient,
                    4
                )
            })

    return {
        "results": results,
        "coefficients": coefficients,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "features": features,
        "target": target
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    total_students = len(df)

    placed = int(
        (df["PlacementStatus"] == 1).sum()
    )

    not_placed = int(
        (df["PlacementStatus"] == 0).sum()
    )

    placement_rate = round(
        placed / total_students * 100,
        2
    )

    average_cgpa = round(
        df["CGPA"].mean(),
        2
    )

    average_attendance = round(
        df["AttendancePercent"].mean(),
        2
    )

    missing_values = int(
        df.isnull().sum().sum()
    )

    return render_template(
        "home.html",

        total_students=total_students,

        columns=len(df.columns),

        placed=placed,

        not_placed=not_placed,

        placement_rate=placement_rate,

        average_cgpa=average_cgpa,

        average_attendance=average_attendance,

        missing_values=missing_values
    )


# ============================================================
# SESSION
# ============================================================

@app.route("/session")
def session():

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    eda_plots = collect_plots([

        existing_plot(
            "Output/plot/cgpa_distribution.png",
            "CGPA Distribution"
        ),

        existing_plot(
            "Output/plot/gender_distribution.png",
            "Gender Distribution"
        ),

        existing_plot(
            "Output/plot/cgpa_vs_placement.png",
            "CGPA vs Placement"
        ),

        existing_plot(
            "Output/plot/cgpa_vs_attendance.png",
            "CGPA vs Attendance"
        ),

        existing_plot(
            "Output/plot/cgpa_internships_pairplot.png",
            "CGPA and Internships"
        ),

        existing_plot(
            "Output/plot/correlation_heatmap.png",
            "Correlation Heatmap"
        )
    ])


    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    regression = calculate_regression_results()


    # --------------------------------------------------------
    # TREE / BOOSTING
    # --------------------------------------------------------

    model_plots = collect_plots([

        existing_plot(
            "src/module3/decision_tree.png",
            "Decision Tree"
        ),

        existing_plot(
            "src/module3/decision_tree_feature_importance.png",
            "Decision Tree Feature Importance"
        ),

        existing_plot(
            "src/module3/random_forest_feature_importance.png",
            "Random Forest Feature Importance"
        ),

        existing_plot(
            "src/module3/gradient_boosting_feature_importance.png",
            "Gradient Boosting Feature Importance"
        )
    ])


    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------

    dbscan_plots = collect_plots([

        existing_plot(
            "src/module4/outputs/dbscan/k_distance_plot.png",
            "DBSCAN K-Distance Plot"
        ),

        existing_plot(
            "src/module4/outputs/dbscan/dbscan_clusters_pca_2d.png",
            "DBSCAN PCA Clusters"
        )
    ])


    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca_plots = collect_plots([

        existing_plot(
            "src/module4/pca/scree_plot.png",
            "PCA Scree Plot"
        ),

        existing_plot(
            "src/module4/pca/cumulative_variance.png",
            "Cumulative Explained Variance"
        ),

        existing_plot(
            "src/module4/pca/loadings_biplot.png",
            "PCA Feature Loadings"
        ),

        existing_plot(
            "src/module4/pca/pca_2d_by_placement.png",
            "2D PCA by Placement"
        )
    ])


    return render_template(

        "session.html",

        eda_plots=eda_plots,

        model_plots=model_plots,

        dbscan_plots=dbscan_plots,

        pca_plots=pca_plots,

        regression=regression
    )


# ============================================================
# SERVE FILES
# ============================================================

@app.route("/files/<path:filename>")
def files(filename):

    return send_from_directory(
        BASE_DIR,
        filename
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )