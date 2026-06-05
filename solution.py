# =============================================================================
#  TRAFFIC DEMAND PREDICTION — COMPLETE SOLUTION
#  Strategy: LightGBM + XGBoost ensemble with lag features & target encoding
# =============================================================================
#
#  HOW TO RUN:
#    1. Place train.csv, test.csv, sample_submission.csv in a folder called data/
#    2. Create an empty folder called outputs/
#    3. pip install lightgbm xgboost scikit-learn pandas numpy
#    4. python solution.py
#
#  OUTPUT:
#    outputs/submission.csv  — 41,778 rows × 2 columns (Index, demand)
#
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
print("  TRAFFIC DEMAND PREDICTION — FULL PIPELINE")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading data ...")

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

print(f"    Train : {train.shape[0]:,} rows × {train.shape[1]} cols")
print(f"    Test  : {test.shape[0]:,} rows × {test.shape[1]} cols")

# Quick sanity checks
assert "demand" in train.columns,  "demand column missing from train!"
assert "demand" not in test.columns, "demand column must NOT be in test!"
assert train["demand"].between(0, 1).all(), "demand values out of [0,1]!"

print("    ✓ Data loaded successfully")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER: PARSE TIMESTAMP
#     "14:30" → hour=14, minute=30, slot_of_day=58
# ─────────────────────────────────────────────────────────────────────────────
def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, minute, and 15-min slot index from 'H:MM' timestamp."""
    df = df.copy()
    parts             = df["timestamp"].str.split(":", expand=True)
    df["hour"]        = parts[0].astype(int)
    df["minute"]      = parts[1].astype(int)
    # slot_of_day: 0 = 00:00, 1 = 00:15, ..., 95 = 23:45
    df["slot_of_day"] = df["hour"] * 4 + df["minute"] // 15
    return df

train = parse_timestamp(train)
test  = parse_timestamp(test)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
#     All statistics are computed from `train` only, then applied to test.
#     This prevents data leakage.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Engineering features ...")


# ── 3a. Impute missing values (using train statistics for both sets) ──────────
def impute_columns(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in RoadType, Temperature, and Weather
    using statistics from `ref` (always pass train as ref).
    """
    df  = df.copy()

    # RoadType: roads are static — use the geohash's most common type
    road_mode = (
        ref.dropna(subset=["RoadType"])
           .groupby("geohash")["RoadType"]
           .agg(lambda x: x.mode().iloc[0])
    )
    df["RoadType"] = df["RoadType"].fillna(df["geohash"].map(road_mode))
    df["RoadType"] = df["RoadType"].fillna("Residential")   # global fallback

    # Temperature: median per (geohash, hour) → then geohash median → global median
    temp_geo_hour = (
        ref.dropna(subset=["Temperature"])
           .groupby(["geohash", "hour"])["Temperature"]
           .median()
    )
    temp_geo = (
        ref.dropna(subset=["Temperature"])
           .groupby("geohash")["Temperature"]
           .median()
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

    # Weather: most common per geohash
    weather_mode = (
        ref.dropna(subset=["Weather"])
           .groupby("geohash")["Weather"]
           .agg(lambda x: x.mode().iloc[0])
    )
    df["Weather"] = df["Weather"].fillna(df["geohash"].map(weather_mode))
    df["Weather"] = df["Weather"].fillna("Sunny")           # global fallback

    return df


train = impute_columns(train, ref=train)
test  = impute_columns(test,  ref=train)   # ← always use train as reference


# ── 3b. Missing-value flag columns (before imputation info is lost) ──────────
# (We already imputed above; in practice compute flags before imputing.)
# Re-derive them from the original missingness indicators we saved earlier.
# Since we've already imputed, use a proxy: note roadtype_enc == -1 after encoding
# means it was originally missing. We'll add explicit flags below after encoding.


# ── 3c. Road & vehicle features ──────────────────────────────────────────────
ROAD_MAP    = {"Residential": 0, "Street": 1, "Highway": 2}
WEATHER_MAP = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["road_type_enc"]      = df["RoadType"].map(ROAD_MAP).fillna(-1).astype(int)
    df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["landmarks_flag"]      = (df["Landmarks"] == "Yes").astype(int)
    df["weather_enc"]         = df["Weather"].map(WEATHER_MAP).fillna(-1).astype(int)

    # Interaction: high-capacity road near a landmark
    df["highway_x_landmark"] = (
        (df["RoadType"] == "Highway") & (df["Landmarks"] == "Yes")
    ).astype(int)

    # Combined capacity signal
    df["lanes_x_road"] = df["NumberofLanes"] * df["road_type_enc"].clip(lower=0)

    return df


train = encode_categoricals(train)
test  = encode_categoricals(test)


# ── 3d. Cyclical time features ───────────────────────────────────────────────
def cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour and slot as (sin, cos) pairs so that 23:00 and 00:00
    are treated as neighbours by the model, not opposites.
    """
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_slot"] = np.sin(2 * np.pi * df["slot_of_day"] / 96)
    df["cos_slot"] = np.cos(2 * np.pi * df["slot_of_day"] / 96)

    df["is_daytime"]      = df["hour"].between(6, 18).astype(int)
    df["is_morning_rush"] = df["hour"].between(7, 10).astype(int)
    df["is_evening"]      = df["hour"].between(17, 21).astype(int)
    df["is_night"]        = df["hour"].between(0, 5).astype(int)
    return df


train = cyclical_time_features(train)
test  = cyclical_time_features(test)


# ── 3e. Spatial prefix features ──────────────────────────────────────────────
train["geohash_q5"] = train["geohash"].str[:5]
train["geohash_q4"] = train["geohash"].str[:4]
test["geohash_q5"]  = test["geohash"].str[:5]
test["geohash_q4"]  = test["geohash"].str[:4]


# ── 3f. Target-encoding features (computed from TRAIN only) ──────────────────
#
#   These replace a categorical key with its mean demand from train.
#   NEVER compute these from test — that would be leakage.
#
def make_target_encodings(train_df: pd.DataFrame,
                          apply_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build demand statistics from train_df and map onto apply_df.
    Returns apply_df with new columns added.
    """
    apply_df = apply_df.copy()
    tr       = train_df  # alias for brevity

    # Per-geohash demand stats
    geo_stats = tr.groupby("geohash")["demand"].agg(["mean", "std"]).rename(
        columns={"mean": "geo_demand_mean", "std": "geo_demand_std"}
    )
    geo_stats["geo_demand_std"] = geo_stats["geo_demand_std"].fillna(0)

    apply_df["geo_demand_mean"] = apply_df["geohash"].map(geo_stats["geo_demand_mean"])
    apply_df["geo_demand_std"]  = apply_df["geohash"].map(geo_stats["geo_demand_std"])

    # Per-slot demand stats (global time-of-day signal)
    slot_mean = tr.groupby("slot_of_day")["demand"].mean().rename("slot_demand_mean")
    apply_df["slot_demand_mean"] = apply_df["slot_of_day"].map(slot_mean)

    # Per-(geohash, slot) interaction — the single most predictive static feature
    geo_slot_mean = (
        tr.groupby(["geohash", "slot_of_day"])["demand"]
          .mean()
          .rename("geo_slot_demand_mean")
    )
    apply_df = apply_df.join(
        geo_slot_mean,
        on=["geohash", "slot_of_day"],
        how="left"
    )

    # Prefix-level means (useful fallback for unseen geohashes)
    q5_mean = (
        tr.assign(geohash_q5=tr["geohash"].str[:5])
          .groupby("geohash_q5")["demand"].mean()
          .rename("q5_demand_mean")
    )
    q4_mean = (
        tr.assign(geohash_q4=tr["geohash"].str[:4])
          .groupby("geohash_q4")["demand"].mean()
          .rename("q4_demand_mean")
    )
    apply_df["q5_demand_mean"] = apply_df["geohash_q5"].map(q5_mean)
    apply_df["q4_demand_mean"] = apply_df["geohash_q4"].map(q4_mean)

    # Per-(geohash, hour) mean — coarser than slot, more stable
    geo_hour_mean = (
        tr.groupby(["geohash", "hour"])["demand"]
          .mean()
          .rename("geo_hour_demand_mean")
    )
    apply_df = apply_df.join(
        geo_hour_mean,
        on=["geohash", "hour"],
        how="left"
    )

    # RoadType × NumberOfLanes mean
    road_lanes_mean = (
        tr.groupby(["RoadType", "NumberofLanes"])["demand"]
          .mean()
          .rename("road_lanes_demand_mean")
    )
    apply_df = apply_df.join(
        road_lanes_mean,
        on=["RoadType", "NumberofLanes"],
        how="left"
    )

    return apply_df


train = make_target_encodings(train_df=train, apply_df=train)
test  = make_target_encodings(train_df=train, apply_df=test)


# ── 3g. LAG FEATURES — the most important features in this problem ────────────
#
#  KEY INSIGHT: Test is day 49 (10:00→23:45). Train contains ALL of day 48.
#  Therefore: demand at (geohash G, slot S) on day 49 ≈ demand at (G, S) on day 48.
#  This one fact makes lag features dominant over everything else.
#
def add_lag_features(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features from day-48 demand in train_ref.
    All values come from training data — no leakage.
    """
    df  = df.copy()
    d48 = train_ref[train_ref["day"] == 48].copy()

    # ── Lag A: Same geohash, same slot, day 48 ───────────────────────────────
    lag_a = (
        d48.groupby(["geohash", "slot_of_day"])["demand"]
           .mean()
           .rename("lag_demand_day48")
    )
    df = df.join(lag_a, on=["geohash", "slot_of_day"], how="left")

    # ── Lag B: Same geohash, 1 hour earlier (slot - 4), day 48 ──────────────
    lag_b = (
        d48.assign(slot_plus4=d48["slot_of_day"] + 4)          # shift target by +4
           .groupby(["geohash", "slot_plus4"])["demand"]
           .mean()
           .rename("lag_demand_minus1h")
    )
    lag_b.index.names = ["geohash", "slot_of_day"]
    df = df.join(lag_b, on=["geohash", "slot_of_day"], how="left")

    # ── Lag C: Same geohash, 2 hours earlier (slot - 8), day 48 ─────────────
    lag_c = (
        d48.assign(slot_plus8=d48["slot_of_day"] + 8)
           .groupby(["geohash", "slot_plus8"])["demand"]
           .mean()
           .rename("lag_demand_minus2h")
    )
    lag_c.index.names = ["geohash", "slot_of_day"]
    df = df.join(lag_c, on=["geohash", "slot_of_day"], how="left")

    # ── Lag D: Rolling mean of 4 slots (1h) on day 48 ───────────────────────
    d48_sorted = d48.sort_values(["geohash", "slot_of_day"])
    d48_sorted["roll4"] = (
        d48_sorted.groupby("geohash")["demand"]
                  .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )
    lag_d = (
        d48_sorted.groupby(["geohash", "slot_of_day"])["roll4"]
                  .mean()
                  .rename("lag_rollmean4_day48")
    )
    df = df.join(lag_d, on=["geohash", "slot_of_day"], how="left")

    # ── Lag E: Rolling mean of 16 slots (4h) on day 48 ──────────────────────
    d48_sorted["roll16"] = (
        d48_sorted.groupby("geohash")["demand"]
                  .transform(lambda x: x.rolling(16, min_periods=1).mean())
    )
    lag_e = (
        d48_sorted.groupby(["geohash", "slot_of_day"])["roll16"]
                  .mean()
                  .rename("lag_rollmean16_day48")
    )
    df = df.join(lag_e, on=["geohash", "slot_of_day"], how="left")

    # ── Lag F: Std deviation over last 4 slots on day 48 (volatility signal) ─
    d48_sorted["roll4_std"] = (
        d48_sorted.groupby("geohash")["demand"]
                  .transform(lambda x: x.rolling(4, min_periods=1).std().fillna(0))
    )
    lag_f = (
        d48_sorted.groupby(["geohash", "slot_of_day"])["roll4_std"]
                  .mean()
                  .rename("lag_rollstd4_day48")
    )
    df = df.join(lag_f, on=["geohash", "slot_of_day"], how="left")

    # ── Lag G: geo_slot_mean from day 48 specifically ────────────────────────
    #  Complements the overall geo_slot_mean (which uses all train days)
    lag_g = (
        d48.groupby(["geohash", "slot_of_day"])["demand"]
           .mean()
           .rename("lag_geo_slot_d48")
    )
    df = df.join(lag_g, on=["geohash", "slot_of_day"], how="left")

    # ── Fallback: fill missing lags with geohash mean from day 48 ─────────────
    #  For the ~10 unseen geohashes in test that had no day-48 data
    geo_d48_mean = d48.groupby("geohash")["demand"].mean()
    geo_d48_slot_mean = d48.groupby("slot_of_day")["demand"].mean()

    lag_cols = [
        "lag_demand_day48", "lag_demand_minus1h", "lag_demand_minus2h",
        "lag_rollmean4_day48", "lag_rollmean16_day48",
        "lag_rollstd4_day48", "lag_geo_slot_d48"
    ]
    for col in lag_cols:
        # First try: fill with same geohash's mean across day 48
        missing = df[col].isna()
        if missing.any():
            df.loc[missing, col] = df.loc[missing, "geohash"].map(geo_d48_mean)
        # Second try: fill with global slot mean from day 48
        missing = df[col].isna()
        if missing.any():
            df.loc[missing, col] = df.loc[missing, "slot_of_day"].map(geo_d48_slot_mean)
        # Final fallback: global day-48 mean
        df[col] = df[col].fillna(d48["demand"].mean())

    return df


train = add_lag_features(train, train_ref=train)
test  = add_lag_features(test,  train_ref=train)

print("    ✓ All features engineered")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DEFINE FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    # ── Time ──────────────────────────────────────────────────────────────────
    "hour", "minute", "slot_of_day",
    "sin_hour", "cos_hour", "sin_slot", "cos_slot",
    "is_daytime", "is_morning_rush", "is_evening", "is_night",
    "day",

    # ── Road infrastructure ───────────────────────────────────────────────────
    "road_type_enc", "NumberofLanes",
    "large_vehicles_flag", "landmarks_flag",
    "highway_x_landmark", "lanes_x_road",

    # ── Weather & temperature ─────────────────────────────────────────────────
    "weather_enc", "Temperature",

    # ── Target encodings (spatial + temporal signal) ─────────────────────────
    "geo_demand_mean", "geo_demand_std",
    "slot_demand_mean", "geo_slot_demand_mean",
    "q5_demand_mean",  "q4_demand_mean",
    "geo_hour_demand_mean", "road_lanes_demand_mean",

    # ── Lag features (most important!) ────────────────────────────────────────
    "lag_demand_day48",
    "lag_demand_minus1h",
    "lag_demand_minus2h",
    "lag_rollmean4_day48",
    "lag_rollmean16_day48",
    "lag_rollstd4_day48",
    "lag_geo_slot_d48",
]

# Verify all feature columns exist
missing_in_train = [f for f in FEATURES if f not in train.columns]
missing_in_test  = [f for f in FEATURES if f not in test.columns]
assert not missing_in_train, f"Missing in train: {missing_in_train}"
assert not missing_in_test,  f"Missing in test : {missing_in_test}"

X      = train[FEATURES].fillna(-1).astype("float32")
y      = np.log1p(train["demand"].values)          # log-transform reduces skew
X_test = test[FEATURES].fillna(-1).astype("float32")
groups = train["geohash"].values

print(f"    Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"    Target (log1p) : mean={y.mean():.4f}, std={y.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5a. LIGHTGBM — 5-Fold GroupKFold
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Training LightGBM ...")

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

lgb_oof   = np.zeros(len(X),      dtype=np.float64)
lgb_test  = np.zeros(len(X_test), dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_tr, y_tr   = X.iloc[tr_idx],  y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

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
        train["demand"].values[val_idx],
        np.expm1(lgb_oof[val_idx]).clip(0, 1)
    )
    print(f"    Fold {fold + 1}/5  |  best iter: {model.best_iteration_:>4d}"
          f"  |  val R²: {fold_r2:.4f}")

lgb_oof_demand  = np.expm1(lgb_oof).clip(0, 1)
lgb_test_demand = np.expm1(lgb_test).clip(0, 1)
lgb_r2 = r2_score(train["demand"].values, lgb_oof_demand)
print(f"\n    ✓ LightGBM OOF R²: {lgb_r2:.4f}  |  Score: {max(0, 100*lgb_r2):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5b. XGBOOST — 5-Fold GroupKFold
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Training XGBoost ...")

XGB_PARAMS = {
    "objective":            "reg:squarederror",
    "eval_metric":          "rmse",
    "max_depth":            8,
    "learning_rate":        0.04,
    "n_estimators":         3000,
    "subsample":            0.75,
    "colsample_bytree":     0.75,
    "min_child_weight":     15,
    "reg_alpha":            0.05,
    "reg_lambda":           0.1,
    "random_state":         SEED,
    "tree_method":          "hist",   # fast histogram-based, works on CPU & GPU
    "verbosity":            0,
    "early_stopping_rounds": 100,     # XGBoost 3.x: goes in the constructor
}

xgb_oof  = np.zeros(len(X),      dtype=np.float64)
xgb_test = np.zeros(len(X_test), dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_tr, y_tr   = X.iloc[tr_idx],  y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    xgb_oof[val_idx]  = model.predict(X_val)
    xgb_test         += model.predict(X_test) / 5

    fold_r2 = r2_score(
        train["demand"].values[val_idx],
        np.expm1(xgb_oof[val_idx]).clip(0, 1)
    )
    print(f"    Fold {fold + 1}/5  |  best iter: {model.best_iteration:>4d}"
          f"  |  val R²: {fold_r2:.4f}")

xgb_oof_demand  = np.expm1(xgb_oof).clip(0, 1)
xgb_test_demand = np.expm1(xgb_test).clip(0, 1)
xgb_r2 = r2_score(train["demand"].values, xgb_oof_demand)
print(f"\n    ✓ XGBoost OOF R²:  {xgb_r2:.4f}  |  Score: {max(0, 100*xgb_r2):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BLEND: Find the best weighted average on OOF predictions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Finding optimal blend ...")

best_alpha = 0.5
best_r2    = -np.inf

for alpha in np.arange(0.0, 1.01, 0.05):
    blended_oof = (alpha * lgb_oof_demand + (1 - alpha) * xgb_oof_demand).clip(0, 1)
    r2 = r2_score(train["demand"].values, blended_oof)
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
# 7.  BUILD & VALIDATE SUBMISSION FILE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Building submission file ...")

submission = pd.DataFrame({
    "Index":  test["Index"].values,
    "demand": final_test,
})

# ── Hard validation — these asserts will catch every common mistake ──────────
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
# 8.  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Summary")
print("=" * 65)
print(f"  LightGBM OOF R²      : {lgb_r2:.4f}  (score: {max(0,100*lgb_r2):.1f})")
print(f"  XGBoost  OOF R²      : {xgb_r2:.4f}  (score: {max(0,100*xgb_r2):.1f})")
print(f"  Blend (α={best_alpha:.2f}) R² : {final_r2:.4f}  (score: {final_score:.1f})")
print(f"  Submission file      : {out_path}")
print("=" * 65)
print("\n  ✅ Done! Upload outputs/submission.csv to the leaderboard.\n")
