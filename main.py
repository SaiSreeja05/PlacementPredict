from flask import Flask, render_template, send_from_directory
import os
import pandas as pd

app = Flask(__name__)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Home / Dashboard
# --------------------------------------------------

@app.route("/")
def index():

    total_students = len(df)

    placed_students = 0

    if "PlacementStatus" in df.columns:
        placed_students = (
            df["PlacementStatus"]
            .astype(str)
            .str.lower()
            .eq("placed")
            .sum()
        )

    not_placed_students = total_students - placed_students

    placement_rate = (
        (placed_students / total_students) * 100
        if total_students > 0
        else 0
    )

    average_cgpa = (
        round(df["CGPA"].mean(), 2)
        if "CGPA" in df.columns
        else 0
    )

    average_attendance = (
        round(df["AttendancePercent"].mean(), 2)
        if "AttendancePercent" in df.columns
        else 0
    )

    return render_template(
        "index.html",
        rows=df.shape[0],
        columns=df.shape[1],
        missing_values=int(df.isnull().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        column_names=df.columns.tolist(),
        placed_students=int(placed_students),
        not_placed_students=int(not_placed_students),
        placement_rate=round(placement_rate, 2),
        average_cgpa=average_cgpa,
        average_attendance=average_attendance
    )


# --------------------------------------------------
# Load Data Page
# --------------------------------------------------

@app.route("/load")
def load_page():

    preview = df.head(10).to_html(
        classes="data-table",
        index=False,
        border=0
    )

    return render_template(
        "load.html",
        dataset_name="placement_predict_50k Dataset (2).csv",
        rows=df.shape[0],
        columns=df.shape[1],
        preview=preview,
        column_names=df.columns.tolist()
    )


# --------------------------------------------------
# EDA Page
# --------------------------------------------------

@app.route("/eda")
def eda_page():

    plots = [
        {
            "file": "cgpa_distribution.png",
            "title": "CGPA Distribution"
        },
        {
            "file": "gender_distribution.png",
            "title": "Gender Distribution"
        },
        {
            "file": "cgpa_vs_placement.png",
            "title": "CGPA vs Placement Status"
        },
        {
            "file": "cgpa_vs_attendance.png",
            "title": "CGPA vs Attendance"
        },
        {
            "file": "cgpa_internships_pairplot.png",
            "title": "CGPA and Internships Relationship"
        },
        {
            "file": "correlation_heatmap.png",
            "title": "Correlation Heatmap"
        }
    ]

    return render_template(
        "eda.html",
        plots=plots
    )


# --------------------------------------------------
# Serve plot images from Output/plot
# --------------------------------------------------

@app.route("/plots/<filename>")
def plot_file(filename):

    return send_from_directory(
        PLOT_PATH,
        filename
    )


# --------------------------------------------------
# Feature Engineering Page
# --------------------------------------------------

@app.route("/feature-engg")
def feature_engg_page():

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return render_template(
        "feature_engg.html",
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        total_features=len(df.columns)
    )


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)