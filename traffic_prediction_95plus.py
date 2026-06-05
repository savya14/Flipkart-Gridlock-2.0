# =============================================================================
# TRAFFIC DEMAND PREDICTION — COMPLETE SOLUTION (v6 — OPTIMIZED FOR 97+)
# Strategy: Advanced ensemble with enhanced features, better hyperparameters
# CV: Temporal split — train on all days except last, validate on last day
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  TRAFFIC DEMAND PREDICTION — OPTIMIZED FOR 97+ (v6)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/11] Loading data ...")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"    Train : {train.shape[0]:,} rows × {train.shape[1]} cols")
print(f"    Test  : {test.shape[0]:,} rows × {test.shape[1]} cols")

assert "demand" in train.columns, "demand column missing from train!"
assert "demand" not in test.columns, "demand column must NOT be in test!"
assert train["demand"].between(0, 1).all(), "demand values out of [0,1]!"

if "day" not in test.columns:
    test["day"] = int(train["day"].max()) + 1

print(f"    Train days : {sorted(train['day'].unique().tolist())}")
print(f"    Test days  : {sorted(test['day'].unique().tolist())}")
print("    ✓ Data loaded successfully")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARSE TIMESTAMP
# ─────────────────────────────────────────────────────────────────────────────
def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)
    df["slot_of_day"] = df["hour"] * 4 + df["minute"] // 15
    return df


train = parse_timestamp(train)
test = parse_timestamp(test)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/11] Engineering advanced features ...")


# ── 3a. Missing-value flags ───────────────────────────────────────────────────
def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flag_missing_temperature"] = df["Temperature"].isna().astype(int)
    df["flag_missing_weather"] = df["Weather"].isna().astype(int)
    df["flag_missing_roadtype"] = df["RoadType"].isna().astype(int)
    return df


train = add_missing_flags(train)
test = add_missing_flags(test)


# ── 3b. Impute missing values ─────────────────────────────────────────────────
def safe_mode(series: pd.Series, default=None):
    m = series.mode()
    if len(m) > 0:
        return m.iloc[0]
    return default


def impute_columns(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # NumberofLanes
    lane_median = ref["NumberofLanes"].median()
    df["NumberofLanes"] = df["NumberofLanes"].fillna(lane_median)

    road_mode = (
        ref.dropna(subset=["RoadType"])
        .groupby("geohash")["RoadType"]
        .agg(lambda x: safe_mode(x, default="Residential"))
    )
    global_road = safe_mode(ref["RoadType"].dropna(), default="Residential")
    df["RoadType"] = df["RoadType"].fillna(df["geohash"].map(road_mode))
    df["RoadType"] = df["RoadType"].fillna(global_road)

    temp_geo_hour = (
        ref.dropna(subset=["Temperature"])
        .groupby(["geohash", "hour"])["Temperature"]
        .median()
    )
    temp_geo = ref.dropna(subset=["Temperature"]).groupby("geohash")["Temperature"].median()
    global_temp = ref["Temperature"].median()

    mask = df["Temperature"].isna()
    if mask.any():
        idx = pd.MultiIndex.from_frame(df.loc[mask, ["geohash", "hour"]])
        fill = temp_geo_hour.reindex(idx).values
        df.loc[mask, "Temperature"] = fill

    mask = df["Temperature"].isna()
    if mask.any():
        df.loc[mask, "Temperature"] = df.loc[mask, "geohash"].map(temp_geo)

    df["Temperature"] = df["Temperature"].fillna(global_temp)

    weather_mode = (
        ref.dropna(subset=["Weather"])
        .groupby("geohash")["Weather"]
        .agg(lambda x: safe_mode(x, default="Sunny"))
    )
    global_weather = safe_mode(ref["Weather"].dropna(), default="Sunny")
    df["Weather"] = df["Weather"].fillna(df["geohash"].map(weather_mode))
    df["Weather"] = df["Weather"].fillna(global_weather)

    return df


train = impute_columns(train, ref=train)
test = impute_columns(test, ref=train)

# ── 3c. Categorical encodings ─────────────────────────────────────────────────
ROAD_MAP = {"Residential": 0, "Street": 1, "Highway": 2}
WEATHER_MAP = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["road_type_enc"] = df["RoadType"].map(ROAD_MAP).fillna(-1).astype(int)
    df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["landmarks_flag"] = (df["Landmarks"] == "Yes").astype(int)
    df["weather_enc"] = df["Weather"].map(WEATHER_MAP).fillna(-1).astype(int)
    df["highway_x_landmark"] = (
            (df["RoadType"] == "Highway") & (df["Landmarks"] == "Yes")
    ).astype(int)
    df["lanes_x_road"] = df["NumberofLanes"] * df["road_type_enc"].clip(lower=0)

    # Additional interaction features
    df["weather_x_temp"] = df["weather_enc"] * df["Temperature"]
    df["lanes_x_vehicles"] = df["NumberofLanes"] * df["large_vehicles_flag"]
    df["road_x_weather"] = df["road_type_enc"] * df["weather_enc"]

    return df


train = encode_categoricals(train)
test = encode_categoricals(test)


# ── 3d. Cyclical time features ────────────────────────────────────────────────
def cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_slot"] = np.sin(2 * np.pi * df["slot_of_day"] / 96)
    df["cos_slot"] = np.cos(2 * np.pi * df["slot_of_day"] / 96)

    # More granular time features
    df["is_daytime"] = df["hour"].between(6, 18).astype(int)
    df["is_morning_rush"] = df["hour"].between(7, 10).astype(int)
    df["is_evening_rush"] = df["hour"].between(17, 20).astype(int)
    df["is_evening"] = df["hour"].between(17, 21).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_peak_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)
    df["is_midday"] = df["hour"].between(11, 14).astype(int)
    df["is_late_night"] = df["hour"].between(22, 23).astype(int)

    return df


train = cyclical_time_features(train)
test = cyclical_time_features(test)

# ── 3e. Spatial prefix features ───────────────────────────────────────────────
train["geohash_q5"] = train["geohash"].str[:5]
train["geohash_q4"] = train["geohash"].str[:4]
train["geohash_q3"] = train["geohash"].str[:3]
test["geohash_q5"] = test["geohash"].str[:5]
test["geohash_q4"] = test["geohash"].str[:4]
test["geohash_q3"] = test["geohash"].str[:3]

# ── 3f. Enhanced Target encoding ─────────────────────────────────────────────
TE_SPECS = [
    (["geohash"], "geo_demand_mean"),
    (["slot_of_day"], "slot_demand_mean"),
    (["geohash", "slot_of_day"], "geo_slot_demand_mean"),
    (["geohash_q5"], "q5_demand_mean"),
    (["geohash_q4"], "q4_demand_mean"),
    (["geohash_q3"], "q3_demand_mean"),
    (["geohash", "hour"], "geo_hour_demand_mean"),
    (["RoadType", "NumberofLanes"], "road_lanes_demand_mean"),
    (["geohash", "is_peak_hour"], "geo_peak_demand_mean"),
    (["Weather", "slot_of_day"], "weather_slot_demand_mean"),
    (["RoadType", "hour"], "road_hour_demand_mean"),
]
TE_SMOOTH = 15.0  # Reduced for more signal


def make_target_encodings(train_df: pd.DataFrame,
                          apply_df: pd.DataFrame) -> pd.DataFrame:
    apply_df = apply_df.copy()
    tr = train_df.copy()

    _te_cols = [spec[1] for spec in TE_SPECS] + ["geo_demand_std", "slot_demand_std"]
    apply_df = apply_df.drop(columns=[c for c in _te_cols if c in apply_df.columns])

    global_mean = tr["demand"].mean()

    for keys, name in TE_SPECS:
        grp = tr.groupby(keys)["demand"].agg(["sum", "count"]).reset_index()
        grp[name] = (grp["sum"] + TE_SMOOTH * global_mean) / (grp["count"] + TE_SMOOTH)
        grp = grp[keys + [name]]
        apply_df = apply_df.merge(grp, on=keys, how="left", sort=False)
        apply_df[name] = apply_df[name].fillna(global_mean)

    # Standard deviations
    geo_std = (
        tr.groupby("geohash")["demand"]
        .std()
        .fillna(0)
        .reset_index()
        .rename(columns={"demand": "geo_demand_std"})
    )
    apply_df = apply_df.merge(geo_std, on="geohash", how="left", sort=False)
    apply_df["geo_demand_std"] = apply_df["geo_demand_std"].fillna(0)

    slot_std = (
        tr.groupby("slot_of_day")["demand"]
        .std()
        .fillna(0)
        .reset_index()
        .rename(columns={"demand": "slot_demand_std"})
    )
    apply_df = apply_df.merge(slot_std, on="slot_of_day", how="left", sort=False)
    apply_df["slot_demand_std"] = apply_df["slot_demand_std"].fillna(0)

    return apply_df


def make_target_encodings_loo(train_df: pd.DataFrame) -> pd.DataFrame:
    out = train_df.copy()
    global_mean = out["demand"].mean()

    for keys, name in TE_SPECS:
        out = out.drop(columns=[name], errors="ignore")

        grp = (
            out.groupby(keys)["demand"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": f"{name}__sum", "count": f"{name}__cnt"})
        )

        out = out.merge(grp, on=keys, how="left", sort=False)

        num = (out[f"{name}__sum"] - out["demand"]) + TE_SMOOTH * global_mean
        den = (out[f"{name}__cnt"] - 1) + TE_SMOOTH

        out[name] = np.where(
            (out[f"{name}__cnt"] - 1) > 0,
            num / den,
            global_mean
        )

        out = out.drop(columns=[f"{name}__sum", f"{name}__cnt"])

    # Standard deviations
    geo_std = (
        out.groupby("geohash")["demand"]
        .std()
        .fillna(0)
        .reset_index()
        .rename(columns={"demand": "geo_demand_std"})
    )
    out = out.merge(geo_std, on="geohash", how="left", sort=False)
    out["geo_demand_std"] = out["geo_demand_std"].fillna(0)

    slot_std = (
        out.groupby("slot_of_day")["demand"]
        .std()
        .fillna(0)
        .reset_index()
        .rename(columns={"demand": "slot_demand_std"})
    )
    out = out.merge(slot_std, on="slot_of_day", how="left", sort=False)
    out["slot_demand_std"] = out["slot_demand_std"].fillna(0)

    return out


# ── 3g. Enhanced Lag features ─────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced leak-safe lag features using previous day
    """
    out = df.copy()

    if "day" not in out.columns:
        next_day = int(train_ref["day"].max()) + 1
        out["day"] = next_day

    base = train_ref[["day", "geohash", "slot_of_day", "demand"]].copy()

    global_mean = float(train_ref["demand"].mean())
    geo_mean = train_ref.groupby("geohash")["demand"].mean()
    slot_mean = train_ref.groupby("slot_of_day")["demand"].mean()

    # Lag A: same slot previous day
    lag_same = base.rename(columns={"demand": "lag_demand_day48"})
    lag_same["day"] = lag_same["day"] + 1
    out = out.merge(lag_same, on=["day", "geohash", "slot_of_day"], how="left", sort=False)

    # Lag B: 30min earlier
    lag_30min = base.rename(columns={"demand": "lag_demand_minus30min"})
    lag_30min["day"] = lag_30min["day"] + 1
    lag_30min["slot_of_day"] = lag_30min["slot_of_day"] + 2
    out = out.merge(lag_30min, on=["day", "geohash", "slot_of_day"], how="left", sort=False)

    # Lag C: 1 hour earlier
    lag_minus1 = base.rename(columns={"demand": "lag_demand_minus1h"})
    lag_minus1["day"] = lag_minus1["day"] + 1
    lag_minus1["slot_of_day"] = lag_minus1["slot_of_day"] + 4
    out = out.merge(lag_minus1, on=["day", "geohash", "slot_of_day"], how="left", sort=False)

    # Lag D: 2 hours earlier
    lag_minus2 = base.rename(columns={"demand": "lag_demand_minus2h"})
    lag_minus2["day"] = lag_minus2["day"] + 1
    lag_minus2["slot_of_day"] = lag_minus2["slot_of_day"] + 8
    out = out.merge(lag_minus2, on=["day", "geohash", "slot_of_day"], how="left", sort=False)

    # Lag E: 3 hours earlier
    lag_minus3 = base.rename(columns={"demand": "lag_demand_minus3h"})
    lag_minus3["day"] = lag_minus3["day"] + 1
    lag_minus3["slot_of_day"] = lag_minus3["slot_of_day"] + 12
    out = out.merge(lag_minus3, on=["day", "geohash", "slot_of_day"], how="left", sort=False)

    # Rolling stats within (geohash, day)
    base_roll = train_ref[["day", "geohash", "slot_of_day", "demand"]].copy()
    base_roll = base_roll.sort_values(["geohash", "day", "slot_of_day"])

    base_roll["lag_rollmean4_tmp"] = base_roll.groupby(["geohash", "day"])["demand"] \
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    base_roll["lag_rollmean8_tmp"] = base_roll.groupby(["geohash", "day"])["demand"] \
        .transform(lambda x: x.rolling(8, min_periods=1).mean())
    base_roll["lag_rollmean16_tmp"] = base_roll.groupby(["geohash", "day"])["demand"] \
        .transform(lambda x: x.rolling(16, min_periods=1).mean())
    base_roll["lag_rollstd4_tmp"] = base_roll.groupby(["geohash", "day"])["demand"] \
        .transform(lambda x: x.rolling(4, min_periods=1).std().fillna(0.0))
    base_roll["lag_rollmax8_tmp"] = base_roll.groupby(["geohash", "day"])["demand"] \
        .transform(lambda x: x.rolling(8, min_periods=1).max())
    base_roll["lag_rollmin8_tmp"] = base_roll.groupby(["geohash", "day"])["demand"] \
        .transform(lambda x: x.rolling(8, min_periods=1).min())

    roll = base_roll.rename(columns={
        "lag_rollmean4_tmp": "lag_rollmean4_day48",
        "lag_rollmean8_tmp": "lag_rollmean8_day48",
        "lag_rollmean16_tmp": "lag_rollmean16_day48",
        "lag_rollstd4_tmp": "lag_rollstd4_day48",
        "lag_rollmax8_tmp": "lag_rollmax8_day48",
        "lag_rollmin8_tmp": "lag_rollmin8_day48",
    })
    roll = roll[["day", "geohash", "slot_of_day",
                 "lag_rollmean4_day48", "lag_rollmean8_day48", "lag_rollmean16_day48",
                 "lag_rollstd4_day48", "lag_rollmax8_day48", "lag_rollmin8_day48"]].copy()

    roll["day"] = roll["day"] + 1
    out = out.merge(roll, on=["day", "geohash", "slot_of_day"], how="left", sort=False)

    out["lag_geo_slot_d48"] = out["lag_demand_day48"]

    # Fill missing lag values
    for col in ["lag_demand_day48", "lag_demand_minus30min", "lag_demand_minus1h",
                "lag_demand_minus2h", "lag_demand_minus3h"]:
        out[col] = out[col].fillna(out["geohash"].map(geo_mean))
        out[col] = out[col].fillna(out["slot_of_day"].map(slot_mean))
        out[col] = out[col].fillna(global_mean)

    for col in ["lag_rollmean4_day48", "lag_rollmean8_day48", "lag_rollmean16_day48",
                "lag_rollstd4_day48", "lag_rollmax8_day48", "lag_rollmin8_day48"]:
        out[col] = out[col].fillna(0.0)

    # Derived lag features
    out["lag_diff_1h"] = out["lag_demand_day48"] - out["lag_demand_minus1h"]
    out["lag_diff_2h"] = out["lag_demand_day48"] - out["lag_demand_minus2h"]
    out["lag_range"] = out["lag_rollmax8_day48"] - out["lag_rollmin8_day48"]

    return out


train = add_lag_features(train, train_ref=train)
test = add_lag_features(test, train_ref=train)

print("    ✓ All advanced features engineered")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE LIST
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "hour", "minute", "slot_of_day",
    "sin_hour", "cos_hour", "sin_slot", "cos_slot",
    "is_daytime", "is_morning_rush", "is_evening_rush", "is_evening", "is_night",
    "is_peak_hour", "is_midday", "is_late_night",
    "road_type_enc", "NumberofLanes",
    "large_vehicles_flag", "landmarks_flag",
    "highway_x_landmark", "lanes_x_road",
    "weather_enc", "Temperature",
    "weather_x_temp", "lanes_x_vehicles", "road_x_weather",
    "flag_missing_temperature", "flag_missing_weather", "flag_missing_roadtype",
    "geo_demand_mean", "geo_demand_std",
    "slot_demand_mean", "slot_demand_std",
    "geo_slot_demand_mean",
    "q5_demand_mean", "q4_demand_mean", "q3_demand_mean",
    "geo_hour_demand_mean", "road_lanes_demand_mean",
    "geo_peak_demand_mean", "weather_slot_demand_mean", "road_hour_demand_mean",
    "lag_demand_day48",
    "lag_demand_minus30min",
    "lag_demand_minus1h",
    "lag_demand_minus2h",
    "lag_demand_minus3h",
    "lag_rollmean4_day48",
    "lag_rollmean8_day48",
    "lag_rollmean16_day48",
    "lag_rollstd4_day48",
    "lag_rollmax8_day48",
    "lag_rollmin8_day48",
    "lag_geo_slot_d48",
    "lag_diff_1h",
    "lag_diff_2h",
    "lag_range",
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/11] Preparing temporal validation split ...")

last_day = int(train["day"].max())
hist = train[train["day"] < last_day].copy()
val = train[train["day"] == last_day].copy()

assert len(hist) > 0, "No historical rows before last day."
assert len(val) > 0, "No rows found for last validation day."

print(f"    Historical train rows : {len(hist):,}")
print(f"    Validation rows       : {len(val):,}")
print(f"    Validation day        : {last_day}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TARGET ENCODINGS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/11] Building target encodings ...")

hist_te = make_target_encodings_loo(hist)
val_te = make_target_encodings(hist, val)
test_te = make_target_encodings(train, test)

X_tr = hist_te[FEATURES].fillna(-1).astype("float32")
X_val = val_te[FEATURES].fillna(-1).astype("float32")
X_test = test_te[FEATURES].fillna(-1).astype("float32")

y_tr = np.log1p(hist_te["demand"].values)
y_val = np.log1p(val_te["demand"].values)

print(f"    Feature matrix : {len(X_tr):,} train rows, {len(X_val):,} val rows × {len(FEATURES)} features")
print(f"    Target (log1p) : mean={y_tr.mean():.4f}, std={y_tr.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. LIGHTGBM — OPTIMIZED
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/11] Training LightGBM (optimized) ...")

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 255,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 10,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "n_estimators": 5000,
    "random_state": SEED,
    "verbose": -1,
    "max_depth": -1,
}

lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(150, verbose=False),
        lgb.log_evaluation(300),
    ],
)

best_iter_lgb = getattr(lgb_model, "best_iteration_", None)
if best_iter_lgb is None or best_iter_lgb <= 0:
    best_iter_lgb = 500

lgb_val_pred = np.expm1(lgb_model.predict(X_val)).clip(0, 1)
lgb_test_pred = np.expm1(lgb_model.predict(X_test)).clip(0, 1)
lgb_val_r2 = r2_score(val_te["demand"].values, lgb_val_pred)
print(f"    best iter: {best_iter_lgb}  |  val R²: {lgb_val_r2:.4f}  |  Score: {max(0, 100 * lgb_val_r2):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. XGBOOST — OPTIMIZED
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/11] Training XGBoost (optimized) ...")

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 8,
    "learning_rate": 0.01,
    "n_estimators": 5000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "random_state": SEED,
    "tree_method": "hist",
    "verbosity": 0,
    "early_stopping_rounds": 150,
}

xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

best_iter_xgb = getattr(xgb_model, "best_iteration", None)
if best_iter_xgb is None or best_iter_xgb < 0:
    best_iter_xgb = 500

xgb_val_pred = np.expm1(xgb_model.predict(X_val)).clip(0, 1)
xgb_test_pred = np.expm1(xgb_model.predict(X_test)).clip(0, 1)
xgb_val_r2 = r2_score(val_te["demand"].values, xgb_val_pred)
print(f"    best iter: {best_iter_xgb}  |  val R²: {xgb_val_r2:.4f}  |  Score: {max(0, 100 * xgb_val_r2):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. CATBOOST-STYLE LIGHTGBM (ALTERNATIVE PARAMS)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/11] Training LightGBM variant 2 ...")

LGB_PARAMS2 = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 127,
    "learning_rate": 0.02,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "min_child_samples": 15,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 5000,
    "random_state": SEED + 1,
    "verbose": -1,
    "max_depth": 10,
}

lgb_model2 = lgb.LGBMRegressor(**LGB_PARAMS2)
lgb_model2.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(150, verbose=False),
        lgb.log_evaluation(300),
    ],
)

best_iter_lgb2 = getattr(lgb_model2, "best_iteration_", None)
if best_iter_lgb2 is None or best_iter_lgb2 <= 0:
    best_iter_lgb2 = 500

lgb2_val_pred = np.expm1(lgb_model2.predict(X_val)).clip(0, 1)
lgb2_test_pred = np.expm1(lgb_model2.predict(X_test)).clip(0, 1)
lgb2_val_r2 = r2_score(val_te["demand"].values, lgb2_val_pred)
print(f"    best iter: {best_iter_lgb2}  |  val R²: {lgb2_val_r2:.4f}  |  Score: {max(0, 100 * lgb2_val_r2):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. STACKED ENSEMBLE WITH RIDGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/11] Training stacked Ridge ensemble ...")

# Create meta-features for validation and test
meta_train = np.column_stack([lgb_val_pred, xgb_val_pred, lgb2_val_pred])
meta_test = np.column_stack([lgb_test_pred, xgb_test_pred, lgb2_test_pred])

# Train Ridge meta-model
ridge = Ridge(alpha=0.1, random_state=SEED)
ridge.fit(meta_train, val_te["demand"].values)

ridge_val_pred = ridge.predict(meta_train).clip(0, 1)
ridge_test_pred = ridge.predict(meta_test).clip(0, 1)
ridge_val_r2 = r2_score(val_te["demand"].values, ridge_val_pred)

print(f"    Ridge weights: LGB={ridge.coef_[0]:.3f}, XGB={ridge.coef_[1]:.3f}, LGB2={ridge.coef_[2]:.3f}")
print(f"    Ridge val R²: {ridge_val_r2:.4f}  |  Score: {max(0, 100 * ridge_val_r2):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. WEIGHTED BLEND OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/11] Finding optimal weighted blend ...")

best_weights = [0.33, 0.33, 0.34]
best_r2 = -np.inf

# Grid search over weights
from itertools import product

for w1 in np.arange(0.2, 0.6, 0.05):
    for w2 in np.arange(0.2, 0.6, 0.05):
        for w3 in np.arange(0.0, 0.4, 0.05):
            if abs(w1 + w2 + w3 - 1.0) < 0.01:
                blended = (w1 * lgb_val_pred + w2 * xgb_val_pred + w3 * lgb2_val_pred).clip(0, 1)
                r2 = r2_score(val_te["demand"].values, blended)
                if r2 > best_r2:
                    best_r2 = r2
                    best_weights = [w1, w2, w3]

blend_test = (best_weights[0] * lgb_test_pred +
              best_weights[1] * xgb_test_pred +
              best_weights[2] * lgb2_test_pred).clip(0, 1)

print(f"    Best weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, LGB2={best_weights[2]:.2f}")
print(f"    Blend val R²: {best_r2:.4f}  |  Score: {max(0, 100 * best_r2):.2f}")

# Choose best between Ridge and weighted blend
if ridge_val_r2 > best_r2:
    print(f"    → Using Ridge ensemble (R²={ridge_val_r2:.4f})")
    final_val_pred = ridge_val_pred
    final_test_pred = ridge_test_pred
    final_r2 = ridge_val_r2
else:
    print(f"    → Using weighted blend (R²={best_r2:.4f})")
    final_val_pred = (best_weights[0] * lgb_val_pred +
                      best_weights[1] * xgb_val_pred +
                      best_weights[2] * lgb2_val_pred).clip(0, 1)
    final_test_pred = blend_test
    final_r2 = best_r2

# ─────────────────────────────────────────────────────────────────────────────
# 12. REFIT ON FULL TRAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10/11] Refitting all models on full train ...")

train_full_te = make_target_encodings_loo(train)
X_full = train_full_te[FEATURES].fillna(-1).astype("float32")
y_full = np.log1p(train_full_te["demand"].values)

# Refit with extended iterations
lgb_final_estimators = max(200, int(best_iter_lgb * 1.3))
xgb_final_estimators = max(200, int((best_iter_xgb + 1) * 1.3))
lgb2_final_estimators = max(200, int(best_iter_lgb2 * 1.3))

print(f"    Final LGB estimators  : {lgb_final_estimators}")
print(f"    Final XGB estimators  : {xgb_final_estimators}")
print(f"    Final LGB2 estimators : {lgb2_final_estimators}")

# LGB 1
LGB_FINAL_PARAMS = dict(LGB_PARAMS)
LGB_FINAL_PARAMS["n_estimators"] = lgb_final_estimators
lgb_final = lgb.LGBMRegressor(**LGB_FINAL_PARAMS)
lgb_final.fit(X_full, y_full)

# XGB
XGB_FINAL_PARAMS = dict(XGB_PARAMS)
XGB_FINAL_PARAMS["n_estimators"] = xgb_final_estimators
XGB_FINAL_PARAMS.pop("early_stopping_rounds", None)
xgb_final = xgb.XGBRegressor(**XGB_FINAL_PARAMS)
xgb_final.fit(X_full, y_full, verbose=False)

# LGB 2
LGB_FINAL_PARAMS2 = dict(LGB_PARAMS2)
LGB_FINAL_PARAMS2["n_estimators"] = lgb2_final_estimators
lgb_final2 = lgb.LGBMRegressor(**LGB_FINAL_PARAMS2)
lgb_final2.fit(X_full, y_full)

# Generate final predictions
lgb_test_final = np.expm1(lgb_final.predict(X_test)).clip(0, 1)
xgb_test_final = np.expm1(xgb_final.predict(X_test)).clip(0, 1)
lgb2_test_final = np.expm1(lgb_final2.predict(X_test)).clip(0, 1)

# Apply same blending strategy
if ridge_val_r2 > best_r2:
    # Refit Ridge on full validation predictions
    meta_test_final = np.column_stack([lgb_test_final, xgb_test_final, lgb2_test_final])
    final_test_refit = ridge.predict(meta_test_final).clip(0, 1)
else:
    final_test_refit = (best_weights[0] * lgb_test_final +
                        best_weights[1] * xgb_test_final +
                        best_weights[2] * lgb2_test_final).clip(0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 13. SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11/11] Building submission file ...")

submission = pd.DataFrame({
    "Index": test["Index"].values,
    "demand": final_test_refit,
})

expected_rows = test.shape[0]
assert submission.shape == (expected_rows, 2), f"❌ Wrong shape: {submission.shape}"
assert list(submission.columns) == ["Index", "demand"], "❌ Wrong column names"
assert submission["Index"].tolist() == test["Index"].tolist(), "❌ Index mismatch"
assert submission["demand"].isna().sum() == 0, "❌ NaN in demand"
assert submission["demand"].between(0, 1).all(), "❌ Predictions out of [0,1]"

out_path = "outputs/submission.csv"
submission.to_csv(out_path, index=False)

print(f"    Shape        : {submission.shape}")
print(f"    Demand mean  : {submission['demand'].mean():.4f}")
print(f"    Demand std   : {submission['demand'].std():.4f}")
print(f"    Demand range : [{submission['demand'].min():.4f}, {submission['demand'].max():.4f}]")
print(f"    ✓ Saved to   : {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY — OPTIMIZED FOR 97+")
print("=" * 65)
print(f"  LightGBM v1 val R²     : {lgb_val_r2:.4f}  (score: {max(0, 100 * lgb_val_r2):.1f})")
print(f"  XGBoost val R²         : {xgb_val_r2:.4f}  (score: {max(0, 100 * xgb_val_r2):.1f})")
print(f"  LightGBM v2 val R²     : {lgb2_val_r2:.4f}  (score: {max(0, 100 * lgb2_val_r2):.1f})")
print(f"  Ridge ensemble R²      : {ridge_val_r2:.4f}  (score: {max(0, 100 * ridge_val_r2):.1f})")
print(f"  Weighted blend R²      : {best_r2:.4f}  (score: {max(0, 100 * best_r2):.1f})")
print(f"  Final validation R²    : {final_r2:.4f}  (score: {max(0, 100 * final_r2):.1f})")
print(f"  Submission file        : {out_path}")
print("=" * 65)
print("\n  ✅ Done! Upload outputs/submission.csv to the leaderboard.\n")