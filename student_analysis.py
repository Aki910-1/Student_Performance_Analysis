"""Student Performance Analysis pipeline.

Run:  python student_analysis.py
To use your own data, skip generate_raw_data() and point RAW_CSV at your file
(it needs: student_id, class, term, subject, marks, attendance_pct, study_hours).
"""
import json
import numpy as np
import pandas as pd

RAW_CSV = "raw_student_data.csv"
CLEAN_CSV = "cleaned_student_data.csv"
TEMPLATE = "dashboard_template.html"
DASHBOARD = "student_dashboard.html"

CLASSES = ["10-A", "10-B", "10-C", "10-D"]
CLASS_EFFECT = {"10-A": 4, "10-B": 1, "10-C": -1, "10-D": -5}
SUBJECTS = ["Mathematics", "Science", "English", "Social Science", "Computer Science"]
SUBJ_DIFF = {"Mathematics": -5, "Science": -2, "English": 3, "Social Science": 2, "Computer Science": 4}
N_PER_CLASS = 40
TERMS = [1, 2, 3]


def generate_raw_data(seed=7):
    rng = np.random.default_rng(seed)
    students = [(sid, cls) for cls in CLASSES for sid in range(1, N_PER_CLASS + 1)]
    students = [(i + 1, cls) for i, (_, cls) in enumerate(students)]

    rows = []
    for sid, cls in students:
        aptitude = rng.normal(65, 10)
        for term in TERMS:
            attendance = np.clip(rng.normal(80, 10) + 0.15 * (aptitude - 65) + (term - 1), 45, 100)
            study_hours = np.clip(rng.normal(8, 3) + 0.05 * (aptitude - 65) + (term - 1) * 0.3, 0, 20)
            for subj in SUBJECTS:
                marks = (aptitude * 0.5 + attendance * 0.35 + study_hours * 1.2 + SUBJ_DIFF[subj]
                         + CLASS_EFFECT[cls] + (term - 1) * 2.0 + rng.normal(0, 6))
                rows.append([sid, cls, term, subj, round(float(np.clip(marks, 0, 100)), 1),
                             round(float(attendance), 1), round(float(study_hours), 1)])
    df = pd.DataFrame(rows, columns=["student_id", "class", "term", "subject",
                                      "marks", "attendance_pct", "study_hours"])

    # Make the data "dirty": missing marks/attendance, plus a few bad attendance entries (data-entry errors)
    n = len(df)
    df.loc[rng.choice(n, size=int(n * .025), replace=False), "marks"] = np.nan
    df.loc[rng.choice(n, size=int(n * .02), replace=False), "attendance_pct"] = np.nan
    bad = rng.choice(n, size=15, replace=False)
    df.loc[bad, "attendance_pct"] = df.loc[bad, "attendance_pct"].fillna(90) + rng.choice([15, -95], size=15)
    return df


def grade(m):
    if m >= 90: return "A"
    if m >= 75: return "B"
    if m >= 60: return "C"
    if m >= 40: return "D"
    return "F"


def preprocess(df):
    report = {"raw_rows": len(df), "null_cells": int(df.isna().sum().sum())}
    bad_mask = (df.attendance_pct < 0) | (df.attendance_pct > 100)
    report["invalid_attendance_fixed"] = int(bad_mask.sum())
    df.loc[bad_mask, "attendance_pct"] = np.nan
    df["marks"] = df.groupby("subject")["marks"].transform(lambda s: s.fillna(s.mean())).round(1)
    df["attendance_pct"] = df.groupby("class")["attendance_pct"].transform(lambda s: s.fillna(s.median())).round(1)
    report["remaining_nulls"] = int(df.isna().sum().sum())
    report["clean_rows"] = len(df)
    df["grade"] = df["marks"].apply(grade)
    return df, report


def analyze(df):
    out = {}
    out["overall"] = {"avg_marks": df.marks.mean(), "avg_attendance": df.attendance_pct.mean(),
                       "avg_study_hours": df.study_hours.mean(), "pass_rate": (df.marks >= 40).mean() * 100}
    out["by_subject"] = df.groupby("subject").marks.mean().sort_values(ascending=False)
    out["by_class"] = df.groupby("class").marks.mean().sort_values(ascending=False)
    out["by_term"] = df.groupby("term").marks.mean()
    out["grade_dist"] = df.grade.value_counts()

    stu = df.groupby(["student_id", "class", "term"], as_index=False).agg(
        marks=("marks", "mean"), attendance_pct=("attendance_pct", "mean"), study_hours=("study_hours", "mean"))
    out["corr_attendance"] = stu.marks.corr(stu.attendance_pct)
    out["corr_hours"] = stu.marks.corr(stu.study_hours)

    stu["att_band"] = pd.cut(stu.attendance_pct, [0, 70, 80, 90, 101], labels=["<70%", "70-79%", "80-89%", "90-100%"], right=False)
    stu["hrs_band"] = pd.cut(stu.study_hours, [0, 4, 8, 12, 100], labels=["<4 hrs", "4-7 hrs", "8-11 hrs", "12+ hrs"], right=False)
    out["by_att_band"] = stu.groupby("att_band", observed=True).marks.mean()
    out["by_hrs_band"] = stu.groupby("hrs_band", observed=True).marks.mean()
    out["weak_combos"] = df.groupby(["class", "subject"]).marks.mean().sort_values().head(5)
    return out


def build_insights(res):
    bys, byc = res["by_subject"], res["by_class"]
    weak_subj, strong_subj = bys.idxmin(), bys.idxmax()
    weak_class, strong_class = byc.idxmin(), byc.idxmax()
    att_gap = res["by_att_band"].iloc[-1] - res["by_att_band"].iloc[0]
    hrs_gap = res["by_hrs_band"].iloc[-1] - res["by_hrs_band"].iloc[0]
    term_change = res["by_term"].iloc[-1] - res["by_term"].iloc[0]
    weak_combo = res["weak_combos"].index[0]
    return [
        f"<b>Attendance matters:</b> students attending 90-100% of classes average {res['by_att_band'].iloc[-1]:.1f} marks, "
        f"versus {res['by_att_band'].iloc[0]:.1f} for those under 70% attendance — a {att_gap:.1f}-point gap "
        f"(correlation {res['corr_attendance']:.2f}). Raising low-attendance students above 80% is the single biggest lever available.",
        f"<b>Study hours matter too:</b> students studying 12+ hours a week average {res['by_hrs_band'].iloc[-1]:.1f} marks "
        f"versus {res['by_hrs_band'].iloc[0]:.1f} for under 4 hours — a {hrs_gap:.1f}-point gap (correlation {res['corr_hours']:.2f}). "
        f"Structured study-hour targets (e.g. 8+ hours/week) look worth setting.",
        f"<b>{weak_subj}</b> is the weakest subject overall ({bys[weak_subj]:.1f} avg) while <b>{strong_subj}</b> is the strongest "
        f"({bys[strong_subj]:.1f} avg) — {weak_subj} would benefit from extra practice sessions or revised teaching materials.",
        f"<b>{weak_class}</b> trails the other sections ({byc[weak_class]:.1f} avg vs {byc[strong_class]:.1f} in {strong_class}). "
        f"Its weakest spot is <b>{weak_combo[1]}</b> ({res['weak_combos'].iloc[0]:.1f} avg) — a targeted intervention there "
        f"would lift the section's overall average.",
        f"<b>Trend is positive:</b> the average moved from {res['by_term'].iloc[0]:.1f} in Term 1 to {res['by_term'].iloc[-1]:.1f} "
        f"in Term {res['by_term'].index[-1]} ({'+' if term_change>=0 else ''}{term_change:.1f} points), suggesting current "
        f"support measures are working and should continue.",
    ]


def build_dashboard(df, insights):
    data = {
        "rows": df[["student_id", "class", "term", "subject", "marks", "attendance_pct", "study_hours"]].values.tolist(),
        "classes": CLASSES, "subjects": SUBJECTS, "terms": TERMS,
        "insights": insights,
        "note": "Sample dataset generated for this project (nulls and invalid entries cleaned). Marks out of 100.",
    }
    html = open(TEMPLATE, encoding="utf-8").read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(DASHBOARD, "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    raw = generate_raw_data()
    raw.to_csv(RAW_CSV, index=False)
    df, rep = preprocess(pd.read_csv(RAW_CSV))
    df.to_csv(CLEAN_CSV, index=False)
    print("PREPROCESSING REPORT", rep)
    res = analyze(df)
    print("\nOVERALL", {k: round(v, 2) for k, v in res["overall"].items()})
    for k in ["by_subject", "by_class", "by_term", "grade_dist", "by_att_band", "by_hrs_band", "weak_combos"]:
        print(f"\n--- {k} ---\n{res[k]}")
    print("\ncorr_attendance", round(res["corr_attendance"], 3), " corr_hours", round(res["corr_hours"], 3))
    insights = build_insights(res)
    for i in insights:
        print("-", i)
    build_dashboard(df, insights)
