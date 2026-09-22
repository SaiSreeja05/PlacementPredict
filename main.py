from flask import Flask, render_template, send_from_directory
import os
import pandas as pd
app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)
PLOT_PATH = os.path.join(
    BASE_DIR,
    "Output",
    "plot"
)
FEATURE_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Output",
    "feature_engineering"
)
MODULE3_PATH = os.path.join(
    BASE_DIR,
    "src",
    "module3"
)

MODULE3_OUTPUT_PATH = os.path.join(
    MODULE3_PATH,
    "outputs"
)

MODULE4_PATH = os.path.join(
    BASE_DIR,
    "src",
    "module4"
)

MODULE4_OUTPUT_PATH = os.path.join(
    MODULE4_PATH,
    "outputs"
)

DBSCAN_PATH = os.path.join(
    MODULE4_OUTPUT_PATH,
    "dbscan"
)

PCA_PATH = os.path.join(
    MODULE4_PATH,
    "pca"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# HELPER FUNCTION
# ============================================================

def make_plot(file_path, title):

    full_path = os.path.join(
        BASE_DIR,
        file_path
    )

    if os.path.exists(full_path):

        return {
            "path": "/output/" + file_path.replace("\\", "/"),
            "title": title
        }

    return None


def get_existing_plots(plot_list):

    return [
        item
        for item in plot_list
        if item is not None
    ]

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    total_students = len(df)


    # --------------------------------------------------------
    # PLACEMENT STATUS
    # --------------------------------------------------------

    if "PlacementStatus" in df.columns:

        placed_students = int(
            (df["PlacementStatus"] == 1).sum()
        )

        not_placed_students = int(
            (df["PlacementStatus"] == 0).sum()
        )

    else:

        placed_students = 0
        not_placed_students = total_students


    # --------------------------------------------------------
    # PLACEMENT RATE
    # --------------------------------------------------------

    placement_rate = (
        round(
            (placed_students / total_students) * 100,
            2
        )
        if total_students > 0
        else 0
    )


    # --------------------------------------------------------
    # AVERAGE CGPA
    # --------------------------------------------------------

    average_cgpa = (
        round(
            df["CGPA"].mean(),
            2
        )
        if "CGPA" in df.columns
        else 0
    )


    # --------------------------------------------------------
    # AVERAGE ATTENDANCE
    # --------------------------------------------------------

    average_attendance = (
        round(
            df["AttendancePercent"].mean(),
            2
        )
        if "AttendancePercent" in df.columns
        else 0
    )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing_values = int(
        df.isnull().sum().sum()
    )


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )


    return render_template(
        "home.html",

        total_students=total_students,

        placed_students=placed_students,

        not_placed_students=not_placed_students,

        placement_rate=placement_rate,

        average_cgpa=average_cgpa,

        average_attendance=average_attendance,

        missing_values=missing_values,

        duplicate_rows=duplicate_rows,

        dataset_shape=(
            f"{df.shape[0]:,} rows × "
            f"{df.shape[1]} columns"
        )
    )


# ============================================================
# COMPLETE ANALYSIS PAGE
# ============================================================

@app.route("/session")
def session():

    # ========================================================
    # EDA
    # ========================================================

    eda_plots = get_existing_plots([

        make_plot(
            "Output/plot/cgpa_distribution.png",
            "CGPA Distribution"
        ),

        make_plot(
            "Output/plot/gender_distribution.png",
            "Gender Distribution"
        ),

        make_plot(
            "Output/plot/cgpa_vs_placement.png",
            "CGPA vs Placement Status"
        ),

        make_plot(
            "Output/plot/cgpa_vs_attendance.png",
            "CGPA vs Attendance"
        ),

        make_plot(
            "Output/plot/cgpa_internships_pairplot.png",
            "CGPA and Internships Relationship"
        ),

        make_plot(
            "Output/plot/correlation_heatmap.png",
            "Correlation Heatmap"
        )

    ])


    # ========================================================
    # FEATURE ENGINEERING / PREPROCESSING
    # ========================================================

    preprocessing_plots = get_existing_plots([

        make_plot(
            "Output/feature_engineering/logistic_scaler_comparison.png",
            "Logistic Regression Scaling Comparison"
        ),

        make_plot(
            "src/feature/logistic_scaler_comparison.png",
            "Logistic Regression Scaling Comparison"
        )

    ])


    # ========================================================
    # DECISION TREE + RANDOM FOREST
    # ========================================================

    tree_plots = get_existing_plots([

        make_plot(
            "src/module3/decision_tree.png",
            "Decision Tree"
        ),

        make_plot(
            "src/module3/decision_tree_feature_importance.png",
            "Decision Tree Feature Importance"
        ),

        make_plot(
            "src/module3/random_forest_feature_importance.png",
            "Random Forest Feature Importance"
        )

    ])


    # ========================================================
    # BOOSTING
    # ========================================================

    boosting_plots = get_existing_plots([

        make_plot(
            "src/module3/gradient_boosting_feature_importance.png",
            "Gradient Boosting Feature Importance"
        ),

        make_plot(
            "src/module3/outputs/adaboost_confusion_matrix.png",
            "AdaBoost Confusion Matrix"
        ),

        make_plot(
            "src/module3/outputs/adaboost_feature_importance.png",
            "AdaBoost Feature Importance"
        ),

        make_plot(
            "outputs/adaboost_confusion_matrix.png",
            "AdaBoost Confusion Matrix"
        ),

        make_plot(
            "outputs/adaboost_feature_importance.png",
            "AdaBoost Feature Importance"
        )

    ])


    # ========================================================
    # PCA
    # ========================================================

    pca_plots = get_existing_plots([

        make_plot(
            "src/module4/pca/scree_plot.png",
            "PCA Scree Plot"
        ),

        make_plot(
            "src/module4/pca/cumulative_variance.png",
            "PCA Cumulative Explained Variance"
        ),

        make_plot(
            "src/module4/pca/loadings_biplot.png",
            "PCA Feature Loadings"
        ),

        make_plot(
            "src/module4/pca/pca_2d_by_placement.png",
            "2D PCA Projection"
        )

    ])


    # ========================================================
    # DBSCAN
    # ========================================================

    dbscan_plots = get_existing_plots([

        make_plot(
            "src/module4/outputs/dbscan/k_distance_plot.png",
            "DBSCAN K-Distance Plot"
        ),

        make_plot(
            "src/module4/outputs/dbscan/dbscan_clusters_pca_2d.png",
            "DBSCAN PCA Clusters"
        )

    ])


    # ========================================================
    # COMBINE CLUSTERING + PCA
    # ========================================================

    clustering_plots = (
        pca_plots +
        dbscan_plots
    )


    # ========================================================
    # CLASSIFICATION RESULTS
    # ========================================================

    classification_results = [

        {
            "name": "Logistic Regression",
            "description":
                "Predicts whether a student is placed or not placed."
        },

        {
            "name": "Decision Tree",
            "description":
                "Tree-based classification model for placement prediction."
        },

        {
            "name": "Random Forest",
            "description":
                "Ensemble of decision trees for placement prediction."
        },

        {
            "name": "AdaBoost",
            "description":
                "Boosting-based classification model."
        },

        {
            "name": "Gradient Boosting",
            "description":
                "Sequential boosting model for placement prediction."
        }

    ]


    # ========================================================
    # REGRESSION RESULTS
    # ========================================================

    regression_results = [

        {
            "name": "Linear Regression",
            "description":
                "Predicts Salary Package using CGPA, aptitude, coding and interview scores."
        },

        {
            "name": "Batch Gradient Descent",
            "description":
                "Linear regression trained using batch gradient descent."
        },

        {
            "name": "Normal Equation",
            "description":
                "Linear regression solved using the normal equation."
        },

        {
            "name": "Scikit-learn Linear Regression",
            "description":
                "Linear regression using the scikit-learn implementation."
        }

    ]


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    model_comparison = [

        {
            "category": "Classification",
            "models":
                "Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting"
        },

        {
            "category": "Regression",
            "models":
                "Batch Gradient Descent, Normal Equation, Scikit-learn Linear Regression"
        },

        {
            "category": "Clustering",
            "models":
                "DBSCAN"
        },

        {
            "category": "Dimensionality Reduction",
            "models":
                "PCA"
        }

    ]


    # ========================================================
    # RENDER SESSION
    # ========================================================

    return render_template(

        "session.html",

        eda_plots=eda_plots,

        preprocessing_plots=preprocessing_plots,

        tree_plots=tree_plots,

        boosting_plots=boosting_plots,

        clustering_plots=clustering_plots,

        classification_results=classification_results,

        regression_results=regression_results,

        model_comparison=model_comparison

    )


# ============================================================
# OUTPUT IMAGE ROUTE
# ============================================================

@app.route("/output/<path:filename>")
def output_file(filename):

    return send_from_directory(
        BASE_DIR,
        filename
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )