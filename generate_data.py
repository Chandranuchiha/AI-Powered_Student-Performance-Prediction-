"""Generate the SYNTHETIC sample dataset (no real students). Run: python generate_data.py"""
import numpy as np, pandas as pd
from config import RANDOM_SEED

def make(n=600, seed=RANDOM_SEED):
    r = np.random.default_rng(seed)
    ability = r.normal(0, 1, n)
    df = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "attendance_pct": np.clip(78 + 9 * ability + r.normal(0, 8, n), 20, 100),
        "study_hours_week": np.clip(10 + 3 * ability + r.normal(0, 3, n), 0, 50),
        "prior_gpa": np.clip(6.8 + 1.0 * ability + r.normal(0, .6, n), 2, 10),
        "assignment_avg": np.clip(68 + 10 * ability + r.normal(0, 7, n), 10, 100),
        "midterm_score": np.clip(62 + 14 * ability + r.normal(0, 9, n), 5, 100),
        "participation_score": np.clip(6 + 1.5 * ability + r.normal(0, 1.2, n), 0, 10),
    }).round(1)
    df["final_score"] = np.clip(
        0.35 * df.midterm_score + 0.25 * df.assignment_avg + 0.15 * df.attendance_pct
        + 3 * df.prior_gpa + 0.4 * df.study_hours_week + r.normal(0, 5, n) - 12, 0, 100).round(1)
    # inject realistic data-quality problems for the quality checks to find
    df.loc[r.choice(n, 15, replace=False), "attendance_pct"] = np.nan
    df.loc[r.choice(n, 10, replace=False), "midterm_score"] = np.nan
    df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)  # duplicates
    return df

if __name__ == "__main__":
    make().to_csv("data/raw/students_synthetic.csv", index=False)
    print("wrote data/raw/students_synthetic.csv")
