"""Central configuration: columns, ranges, seeds."""
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2          # fraction of the remaining train data
PASS_MARK = 50          # final_score >= PASS_MARK -> passed
AT_RISK_THRESHOLD = 0.5 # predicted pass probability below this -> flagged

ID_COL = "student_id"
TARGET_SCORE = "final_score"
TARGET = "passed"

# column -> (min, max) for numeric fields
FEATURE_RANGES = {
    "attendance_pct": (0, 100),
    "study_hours_week": (0, 80),
    "prior_gpa": (0, 10),
    "assignment_avg": (0, 100),
    "midterm_score": (0, 100),
    "participation_score": (0, 10),
}
FEATURES = list(FEATURE_RANGES)
REQUIRED_COLUMNS = [ID_COL] + FEATURES + [TARGET_SCORE]
