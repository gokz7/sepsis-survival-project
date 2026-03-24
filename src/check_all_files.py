import pandas as pd                                    # for loading data

# ── Load all 3 files ──────────────────────────────────────────────────────────

primary    = pd.read_csv('data/sepsis.csv')            # primary cohort - will be used for training
study      = pd.read_csv('data/s41598-020-73558-3_sepsis_survival_study_cohort.csv')       # study cohort - validation
validation = pd.read_csv('data/s41598-020-73558-3_sepsis_survival_validation_cohort.csv')  # validation cohort - testing

# ── Check shapes ──────────────────────────────────────────────────────────────

print("=== Dataset Shapes ===")
print(f"Primary    (train)      : {primary.shape}")
print(f"Study      (validation) : {study.shape}")
print(f"Validation (test)       : {validation.shape}")

# ── Check columns ─────────────────────────────────────────────────────────────

print("\n=== Column Names ===")
print(f"Primary    : {primary.columns.tolist()}")
print(f"Study      : {study.columns.tolist()}")
print(f"Validation : {validation.columns.tolist()}")

# ── Check if columns match ────────────────────────────────────────────────────

print("\n=== Column Match Check ===")
if primary.columns.tolist() == study.columns.tolist() == validation.columns.tolist():
    print("✅ All 3 files have identical columns - safe to proceed")
else:
    print("❌ Column mismatch detected - need to fix before training")

# ── Check class distribution in each file ────────────────────────────────────

print("\n=== Class Distribution ===")
target = 'hospital_outcome_1alive_0dead'

for name, df in [('Primary', primary), ('Study', study), ('Validation', validation)]:
    alive = (df[target]==1).sum()
    dead  = (df[target]==0).sum()
    print(f"{name:<12} - Alive: {alive:,} ({alive/len(df)*100:.1f}%)  Dead: {dead:,} ({dead/len(df)*100:.1f}%)")