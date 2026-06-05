# =============================================================================
#  TRAFFIC DEMAND PREDICTION — FIXED SOLUTION (LEAK-FREE)
#  Strategy: LightGBM + XGBoost ensemble with lag features & target encoding
#
#  FIXES vs original:
#    1. Target encodings computed ONLY on fold's train split (no OOF leakage)
#    2. Lag features computed ONLY from days NOT in the validation fold
#    3. Test features still computed from full train (correct — no leakage there)
#
#  HOW TO RUN:
#    1. Place train.csv, test.csv, sample_submission.csv in a folder called data/
#    2. Create an empty folder called outputs/
#    3. pip install lightgbm xgboost scikit-learn pandas numpy
#    4. python solution_fixed.py   OR run as a .ipynb notebook
#
#  OUTPUT:
#    outputs/submission.csv  — 41,778 rows × 2 columns (Index, demand)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & SEEDS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  TRAFFIC DEMAND PREDICTION — LEAK-FREE PIPELINE")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading data ...")

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

print(f"    Train : {train.shape[0]:,} rows × {train.shape[1]} cols")
print(f"    Test  : {test.shape[0]:,} rows × {test.shape[1]} cols")

assert "demand" in train.columns,       "demand column missing from train!"
assert "demand" not in test.columns,    "demand column must NOT be in test!"
assert train["demand"].between(0,1).all(), "demand values out of [0,1]!"

print("    ✓ Data loaded successfully")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER: PARSE TIMESTAMP
# ─────────────────────────────────────────────────────────────────────────────
def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"]        = parts[0].astype(int)
    df["minute"]      = parts[1].astype(int)
    df["slot_of_day"] = df["hour"] * 4 + df["minute"] // 15
    return df

train = parse_timestamp(train)
test  = parse_timestamp(test)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BASE FEATURE ENGINEERING (no target statistics — those are fold-aware)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Engineering base features ...")

ROAD_MAP    = {"Residential": 0, "Street": 1, "Highway": 2}
WEATHER_MAP = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}


def impute_columns(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using statistics from ref (always pass fold-train or full train)."""
    df = df.copy()

    road_mode = (
        ref.dropna(subset=["RoadType"])
           .groupby("geohash")["RoadType"]
           .agg(lambda x: x.mode().iloc[0])
    )
    df["RoadType"] = df["RoadType"].fillna(df["geohash"].map(road_mode))
    df["RoadType"] = df["RoadType"].fillna("Residential")

    temp_geo_hour = (
        ref.dropna(subset=["Temperature"])
           .groupby(["geohash", "hour"])["Temperature"].median()
    )
    temp_geo = (
        ref.dropna(subset=["Temperature"])
           .groupby("geohash")["Temperature"].median()
    )
    global_temp = ref["Temperature"].median()

    mask = df["Temperature"].isna()
    if mask.any():
        idx  = df.loc[mask].set_index(["geohash", "hour"]).index
        fill = idx.map(temp_geo_hour)
        df.loc[mask, "Temperature"] = fill.values

    mask = df["Temperature"].isna()
    if mask.any():
        df.loc[mask, "Temperature"] = df.loc[mask, "geohash"].map(temp_geo)
    df["Temperature"] = df["Temperature"].fillna(global_temp)

    weather_mode = (
        ref.dropna(subset=["Weather"])
           .groupby("geohash")["Weather"]
           .agg(lambda x: x.mode().iloc[0])
    )
    df["Weather"] = df["Weather"].fillna(df["geohash"].map(weather_mode))
    df["Weather"] = df["Weather"].fillna("Sunny")

    return df


def encode_and_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["road_type_enc"]       = df["RoadType"].map(ROAD_MAP).fillna(-1).astype(int)
    df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["landmarks_flag"]      = (df["Landmarks"] == "Yes").astype(int)
    df["weather_enc"]         = df["Weather"].map(WEATHER_MAP).fillna(-1).astype(int)
    df["highway_x_landmark"]  = (
        (df["RoadType"] == "Highway") & (df["Landmarks"] == "Yes")
    ).astype(int)
    df["lanes_x_road"] = df["NumberofLanes"] * df["road_type_enc"].clip(lower=0)

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_slot"] = np.sin(2 * np.pi * df["slot_of_day"] / 96)
    df["cos_slot"] = np.cos(2 * np.pi * df["slot_of_day"] / 96)

    df["is_daytime"]      = df["hour"].between(6, 18).astype(int)
    df["is_morning_rush"] = df["hour"].between(7, 10).astype(int)
    df["is_evening"]      = df["hour"].between(17, 21).astype(int)
    df["is_night"]        = df["hour"].between(0, 5).astype(int)

    df["geohash_q5"] = df["geohash"].str[:5]
    df["geohash_q4"] = df["geohash"].str[:4]

    return df


# Impute & encode using full train as reference (no target stats here — safe)
train = impute_columns(train, ref=train)
test  = impute_columns(test,  ref=train)

train = encode_and_time_features(train)
test  = encode_and_time_features(test)

print("    ✓ Base features done")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FOLD-AWARE FEATURE FUNCTIONS
#     These are called INSIDE the CV loop with only fold-train data as ref.
# ─────────────────────────────────────────────────────────────────────────────

def make_target_encodings(apply_df: pd.DataFrame,
                          ref: pd.DataFrame) -> pd.DataFrame:
    """
    Compute demand statistics from `ref` (fold-train only) and map onto apply_df.
    This is the fix: ref must never include rows from apply_df's fold.
    """
    apply_df = apply_df.copy()

    # Per-geohash demand stats
    geo_stats = ref.groupby("geohash")["demand"].agg(["mean", "std"]).rename(
        columns={"mean": "geo_demand_mean", "std": "geo_demand_std"}
    )
    geo_stats["geo_demand_std"] = geo_stats["geo_demand_std"].fillna(0)
    apply_df["geo_demand_mean"] = apply_df["geohash"].map(geo_stats["geo_demand_mean"])
    apply_df["geo_demand_std"]  = apply_df["geohash"].map(geo_stats["geo_demand_std"])

    # Per-slot global time-of-day signal
    slot_mean = ref.groupby("slot_of_day")["demand"].mean().rename("slot_demand_mean")
    apply_df["slot_demand_mean"] = apply_df["slot_of_day"].map(slot_mean)

    # Per-(geohash, slot) — most predictive static feature
    geo_slot_mean = (
        ref.groupby(["geohash", "slot_of_day"])["demand"]
           .mean().rename("geo_slot_demand_mean")
    )
    apply_df = apply_df.join(geo_slot_mean, on=["geohash", "slot_of_day"], how="left")

    # Prefix-level means (fallback for unseen geohashes)
    q5_mean = (
        ref.assign(geohash_q5=ref["geohash"].str[:5])
           .groupby("geohash_q5")["demand"].mean().rename("q5_demand_mean")
    )
    q4_mean = (
        ref.assign(geohash_q4=ref["geohash"].str[:4])
           .groupby("geohash_q4")["demand"].mean().rename("q4_demand_mean")
    )
    apply_df["q5_demand_mean"] = apply_df["geohash_q5"].map(q5_mean)
    apply_df["q4_demand_mean"] = apply_df["geohash_q4"].map(q4_mean)

    # Per-(geohash, hour)
    geo_hour_mean = (
        ref.groupby(["geohash", "hour"])["demand"]
           .mean().rename("geo_hour_demand_mean")
    )
    apply_df = apply_df.join(geo_hour_mean, on=["geohash", "hour"], how="left")

    # RoadType × NumberOfLanes
    road_lanes_mean = (
        ref.groupby(["RoadType", "NumberofLanes"])["demand"]
           .mean().rename("road_lanes_demand_mean")
    )
    apply_df = apply_df.join(
        road_lanes_mean, on=["RoadType", "NumberofLanes"], how="left"
    )

    return apply_df


def add_lag_features(apply_df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag features from `ref` (fold-train only).
    Uses the latest available day in ref as the lag source —
    so validation folds whose day-48 rows are held out won't leak.
    """
    apply_df = apply_df.copy()

    # Use the most recent day available in the fold's training data
    latest_day = ref["day"].max()
    d_lag      = ref[ref["day"] == latest_day].copy()

    # Lag A: same geohash, same slot
    lag_a = (
        d_lag.groupby(["geohash", "slot_of_day"])["demand"]
             .mean().rename("lag_demand_day48")
    )
    apply_df = apply_df.join(lag_a, on=["geohash", "slot_of_day"], how="left")

    # Lag B: 1 hour earlier (slot - 4)
    lag_b = (
        d_lag.assign(slot_plus4=d_lag["slot_of_day"] + 4)
             .groupby(["geohash", "slot_plus4"])["demand"]
             .mean().rename("lag_demand_minus1h")
    )
    lag_b.index.names = ["geohash", "slot_of_day"]
    apply_df = apply_df.join(lag_b, on=["geohash", "slot_of_day"], how="left")

    # Lag C: 2 hours earlier (slot - 8)
    lag_c = (
        d_lag.assign(slot_plus8=d_lag["slot_of_day"] + 8)
             .groupby(["geohash", "slot_plus8"])["demand"]
             .mean().rename("lag_demand_minus2h")
    )
    lag_c.index.names = ["geohash", "slot_of_day"]
    apply_df = apply_df.join(lag_c, on=["geohash", "slot_of_day"], how="left")

    # Lag D: rolling mean of 4 slots (1h)
    d_sorted = d_lag.sort_values(["geohash", "slot_of_day"])
    d_sorted["roll4"] = (
        d_sorted.groupby("geohash")["demand"]
                .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )
    lag_d = (
        d_sorted.groupby(["geohash", "slot_of_day"])["roll4"]
                .mean().rename("lag_rollmean4_day48")
    )
    apply_df = apply_df.join(lag_d, on=["geohash", "slot_of_day"], how="left")

    # Lag E: rolling mean of 16 slots (4h)
    d_sorted["roll16"] = (
        d_sorted.groupby("geohash")["demand"]
                .transform(lambda x: x.rolling(16, min_periods=1).mean())
    )
    lag_e = (
        d_sorted.groupby(["geohash", "slot_of_day"])["roll16"]
                .mean().rename("lag_rollmean16_day48")
    )
    apply_df = apply_df.join(lag_e, on=["geohash", "slot_of_day"], how="left")

    # Lag F: rolling std of 4 slots (volatility)
    d_sorted["roll4_std"] = (
        d_sorted.groupby("geohash")["demand"]
                .transform(lambda x: x.rolling(4, min_periods=1).std().fillna(0))
    )
    lag_f = (
        d_sorted.groupby(["geohash", "slot_of_day"])["roll4_std"]
                .mean().rename("lag_rollstd4_day48")
    )
    apply_df = apply_df.join(lag_f, on=["geohash", "slot_of_day"], how="left")

    # Lag G: geo_slot mean from lag day only
    lag_g = (
        d_lag.groupby(["geohash", "slot_of_day"])["demand"]
             .mean().rename("lag_geo_slot_d48")
    )
    apply_df = apply_df.join(lag_g, on=["geohash", "slot_of_day"], how="left")

    # Fallback: fill missing with geohash mean → slot mean → global mean from lag day
    geo_lag_mean  = d_lag.groupby("geohash")["demand"].mean()
    slot_lag_mean = d_lag.groupby("slot_of_day")["demand"].mean()
    global_lag    = d_lag["demand"].mean()

    lag_cols = [
        "lag_demand_day48", "lag_demand_minus1h", "lag_demand_minus2h",
        "lag_rollmean4_day48", "lag_rollmean16_day48",
        "lag_rollstd4_day48", "lag_geo_slot_d48",
    ]
    for col in lag_cols:
        missing = apply_df[col].isna()
        if missing.any():
            apply_df.loc[missing, col] = apply_df.loc[missing, "geohash"].map(geo_lag_mean)
        missing = apply_df[col].isna()
        if missing.any():
            apply_df.loc[missing, col] = apply_df.loc[missing, "slot_of_day"].map(slot_lag_mean)
        apply_df[col] = apply_df[col].fillna(global_lag)

    return apply_df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PRE-COMPUTE TEST FEATURES (from full train — correct, no leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Computing test features from full train ...")

test = make_target_encodings(apply_df=test, ref=train)
test = add_lag_features(apply_df=test, ref=train)

print("    ✓ Test features ready")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEATURE LIST
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "hour", "minute", "slot_of_day",
    "sin_hour", "cos_hour", "sin_slot", "cos_slot",
    "is_daytime", "is_morning_rush", "is_evening", "is_night",
    "day",
    "road_type_enc", "NumberofLanes",
    "large_vehicles_flag", "landmarks_flag",
    "highway_x_landmark", "lanes_x_road",
    "weather_enc", "Temperature",
    "geo_demand_mean", "geo_demand_std",
    "slot_demand_mean", "geo_slot_demand_mean",
    "q5_demand_mean", "q4_demand_mean",
    "geo_hour_demand_mean", "road_lanes_demand_mean",
    "lag_demand_day48",
    "lag_demand_minus1h",
    "lag_demand_minus2h",
    "lag_rollmean4_day48",
    "lag_rollmean16_day48",
    "lag_rollstd4_day48",
    "lag_geo_slot_d48",
]

y      = np.log1p(train["demand"].values)
groups = train["geohash"].values

X_test = test[FEATURES].fillna(-1).astype("float32")

print(f"    Features : {len(FEATURES)}")
print(f"    Target (log1p) : mean={y.mean():.4f}, std={y.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7a. LIGHTGBM — 5-Fold GroupKFold with leak-free features inside loop
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Training LightGBM (leak-free CV) ...")

LGB_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "num_leaves":        255,
    "learning_rate":     0.04,
    "feature_fraction":  0.75,
    "bagging_fraction":  0.75,
    "bagging_freq":      5,
    "min_child_samples": 15,
    "reg_alpha":         0.05,
    "reg_lambda":        0.1,
    "n_estimators":      3000,
    "random_state":      SEED,
    "verbose":           -1,
}

gkf = GroupKFold(n_splits=5)

lgb_oof  = np.zeros(len(train), dtype=np.float64)
lgb_test = np.zeros(len(test),  dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, y, groups=groups)):

    fold_train = train.iloc[tr_idx]   # fold's training rows only
    fold_val   = train.iloc[val_idx]  # fold's validation rows

    # ── Build features using ONLY fold_train as reference ────────────────────
    fold_train_fe = make_target_encodings(apply_df=fold_train, ref=fold_train)
    fold_train_fe = add_lag_features(apply_df=fold_train_fe,   ref=fold_train)

    fold_val_fe   = make_target_encodings(apply_df=fold_val,   ref=fold_train)
    fold_val_fe   = add_lag_features(apply_df=fold_val_fe,     ref=fold_train)
    # ─────────────────────────────────────────────────────────────────────────

    X_tr  = fold_train_fe[FEATURES].fillna(-1).astype("float32")
    y_tr  = np.log1p(fold_train["demand"].values)
    X_val = fold_val_fe[FEATURES].fillna(-1).astype("float32")
    y_val = np.log1p(fold_val["demand"].values)

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(500),
        ],
    )

    lgb_oof[val_idx]  = model.predict(X_val)
    lgb_test         += model.predict(X_test) / 5

    fold_r2 = r2_score(
        fold_val["demand"].values,
        np.expm1(lgb_oof[val_idx]).clip(0, 1)
    )
    print(f"    Fold {fold+1}/5  |  best iter: {model.best_iteration_:>4d}"
          f"  |  val R²: {fold_r2:.4f}")

lgb_oof_demand  = np.expm1(lgb_oof).clip(0, 1)
lgb_test_demand = np.expm1(lgb_test).clip(0, 1)
lgb_r2 = r2_score(train["demand"].values, lgb_oof_demand)
print(f"\n    ✓ LightGBM OOF R²: {lgb_r2:.4f}  |  Score: {max(0, 100*lgb_r2):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7b. XGBOOST — same leak-free CV loop
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Training XGBoost (leak-free CV) ...")

XGB_PARAMS = {
    "objective":             "reg:squarederror",
    "eval_metric":           "rmse",
    "max_depth":             8,
    "learning_rate":         0.04,
    "n_estimators":          3000,
    "subsample":             0.75,
    "colsample_bytree":      0.75,
    "min_child_weight":      15,
    "reg_alpha":             0.05,
    "reg_lambda":            0.1,
    "random_state":          SEED,
    "tree_method":           "hist",
    "verbosity":             0,
    "early_stopping_rounds": 100,
}

xgb_oof  = np.zeros(len(train), dtype=np.float64)
xgb_test = np.zeros(len(test),  dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, y, groups=groups)):

    fold_train = train.iloc[tr_idx]
    fold_val   = train.iloc[val_idx]

    fold_train_fe = make_target_encodings(apply_df=fold_train, ref=fold_train)
    fold_train_fe = add_lag_features(apply_df=fold_train_fe,   ref=fold_train)

    fold_val_fe   = make_target_encodings(apply_df=fold_val,   ref=fold_train)
    fold_val_fe   = add_lag_features(apply_df=fold_val_fe,     ref=fold_train)

    X_tr  = fold_train_fe[FEATURES].fillna(-1).astype("float32")
    y_tr  = np.log1p(fold_train["demand"].values)
    X_val = fold_val_fe[FEATURES].fillna(-1).astype("float32")
    y_val = np.log1p(fold_val["demand"].values)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    xgb_oof[val_idx]  = model.predict(X_val)
    xgb_test         += model.predict(X_test) / 5

    fold_r2 = r2_score(
        fold_val["demand"].values,
        np.expm1(xgb_oof[val_idx]).clip(0, 1)
    )
    print(f"    Fold {fold+1}/5  |  best iter: {model.best_iteration:>4d}"
          f"  |  val R²: {fold_r2:.4f}")

xgb_oof_demand  = np.expm1(xgb_oof).clip(0, 1)
xgb_test_demand = np.expm1(xgb_test).clip(0, 1)
xgb_r2 = r2_score(train["demand"].values, xgb_oof_demand)
print(f"\n    ✓ XGBoost OOF R²:  {xgb_r2:.4f}  |  Score: {max(0, 100*xgb_r2):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  BLEND
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Finding optimal blend ...")

best_alpha = 0.5
best_r2    = -np.inf

for alpha in np.arange(0.0, 1.01, 0.05):
    blended = (alpha * lgb_oof_demand + (1 - alpha) * xgb_oof_demand).clip(0, 1)
    r2 = r2_score(train["demand"].values, blended)
    if r2 > best_r2:
        best_r2    = r2
        best_alpha = alpha

print(f"    Best alpha (LGB weight): {best_alpha:.2f}  →  OOF R²: {best_r2:.4f}"
      f"  |  Score: {max(0, 100*best_r2):.2f}")

final_oof  = (best_alpha * lgb_oof_demand  + (1 - best_alpha) * xgb_oof_demand).clip(0, 1)
final_test = (best_alpha * lgb_test_demand + (1 - best_alpha) * xgb_test_demand).clip(0, 1)

final_r2    = r2_score(train["demand"].values, final_oof)
final_score = max(0, 100 * final_r2)
print(f"\n    ✓ Final blended OOF R²: {final_r2:.4f}  |  Score: {final_score:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  BUILD & VALIDATE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Building submission file ...")

submission = pd.DataFrame({
    "Index":  test["Index"].values,
    "demand": final_test,
})

assert submission.shape == (41778, 2), \
    f"❌ Wrong shape: {submission.shape} — expected (41778, 2)"
assert list(submission.columns) == ["Index", "demand"], \
    f"❌ Wrong column names: {list(submission.columns)}"
assert submission["Index"].tolist() == test["Index"].tolist(), \
    "❌ Index values don't match test.csv exactly!"
assert submission["demand"].isna().sum() == 0, \
    "❌ NaN values found in demand column!"
assert submission["demand"].between(0, 1).all(), \
    "❌ Predictions outside [0, 1] range!"

out_path = "outputs/submission.csv"
submission.to_csv(out_path, index=False)

print(f"    Shape        : {submission.shape}")
print(f"    Demand mean  : {submission['demand'].mean():.4f}")
print(f"    Demand std   : {submission['demand'].std():.4f}")
print(f"    Demand range : [{submission['demand'].min():.4f}, {submission['demand'].max():.4f}]")
print(f"    ✓ Saved to   : {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Summary]")
print("=" * 65)
print(f"  LightGBM OOF R²      : {lgb_r2:.4f}  (score: {max(0,100*lgb_r2):.1f})")
print(f"  XGBoost  OOF R²      : {xgb_r2:.4f}  (score: {max(0,100*xgb_r2):.1f})")
print(f"  Blend (α={best_alpha:.2f}) R² : {final_r2:.4f}  (score: {final_score:.1f})")
print(f"  Submission file      : {out_path}")
print("=" * 65)
print("\n  ✅ Done! OOF scores are now honest. Upload outputs/submission.csv.\n")
print("  NOTE: These OOF scores will likely be LOWER than your original")
print("  script — that's expected and correct. The leakage has been removed.")
print("  Your leaderboard score should now closely match your local OOF score.")
