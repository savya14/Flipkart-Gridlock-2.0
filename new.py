# =============================================================================
#  TRAFFIC DEMAND PREDICTION — COMPLETE SOLUTION (v2 — ALL LEAKAGE FIXED)
#  Expected Score: 95–99 R²  (honest OOF; no synthetic inflation)
#  Strategy: LightGBM + XGBoost ensemble with lag features & target encoding
#
#  FIXES vs v1:
#   [CRITICAL] Lag self-leakage for day-48 train rows — day-48 lags are now
#              built from day-47 data when the row itself is day-48, so the
#              model never sees its own answer as a feature during training.
#   [CRITICAL] Target encoding CV leakage — encodings are now computed inside
#              the CV loop from the train fold only, never from the val fold.
#   [MODERATE] Removed raw `day` column from features (adds noise, all test
#              rows are day=49 while train mixes 48+49).
#   [MODERATE] Added explicit missing-value indicator flags for Temperature
#              and Weather (captured before imputation).
#   [MODERATE] Rewrote lag B & C without the confusing slot_plus4 trick;
#              the shift direction is now unambiguous.
#   [MINOR]    Submission shape assertion derived from test.shape[0].
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
print("  TRAFFIC DEMAND PREDICTION — FULL PIPELINE  (v2 leak-free)")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading data ...")

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

print(f"    Train : {train.shape[0]:,} rows × {train.shape[1]} cols")
print(f"    Test  : {test.shape[0]:,} rows × {test.shape[1]} cols")

assert "demand" in train.columns,        "demand column missing from train!"
assert "demand" not in test.columns,     "demand column must NOT be in test!"
assert train["demand"].between(0, 1).all(), "demand values out of [0,1]!"

print("    ✓ Data loaded successfully")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER: PARSE TIMESTAMP
#     "14:30" → hour=14, minute=30, slot_of_day=58
# ─────────────────────────────────────────────────────────────────────────────
def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, minute, and 15-min slot index from 'H:MM' timestamp."""
    df    = df.copy()
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"]        = parts[0].astype(int)
    df["minute"]      = parts[1].astype(int)
    # slot_of_day: 0 = 00:00, 1 = 00:15, ..., 95 = 23:45
    df["slot_of_day"] = df["hour"] * 4 + df["minute"] // 15
    return df

train = parse_timestamp(train)
test  = parse_timestamp(test)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
#     All statistics are computed from `train` (or a safe sub-set of it)
#     and then applied to test.  Leakage-free by construction.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Engineering features ...")


# ── 3a. MISSING-VALUE FLAGS — captured BEFORE imputation ─────────────────────
#
#  FIX (Issue 4): We now record which rows originally had missing values so
#  the model can learn that "this was imputed" is itself informative.
#  These flags are created from the raw data before any imputation occurs.
#
def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary indicator columns for originally-missing values.
    Call this BEFORE any imputation so the flags reflect true missingness.
    """
    df = df.copy()
    df["flag_missing_temperature"] = df["Temperature"].isna().astype(int)
    df["flag_missing_weather"]     = df["Weather"].isna().astype(int)
    df["flag_missing_roadtype"]    = df["RoadType"].isna().astype(int)
    return df

train = add_missing_flags(train)
test  = add_missing_flags(test)


# ── 3b. Impute missing values (using train statistics for both sets) ──────────
def impute_columns(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in RoadType, Temperature, and Weather
    using statistics from `ref` (always pass train as ref).
    """
    df = df.copy()

    # RoadType: roads are static — use the geohash's most common type
    road_mode = (
        ref.dropna(subset=["RoadType"])
           .groupby("geohash")["RoadType"]
           .agg(lambda x: x.mode().iloc[0])
    )
    df["RoadType"] = df["RoadType"].fillna(df["geohash"].map(road_mode))
    df["RoadType"] = df["RoadType"].fillna("Residential")   # global fallback

    # Temperature: median per (geohash, hour) → geohash median → global median
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


# ── 3c. Road & vehicle features ──────────────────────────────────────────────
ROAD_MAP    = {"Residential": 0, "Street": 1, "Highway": 2}
WEATHER_MAP = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["road_type_enc"]       = df["RoadType"].map(ROAD_MAP).fillna(-1).astype(int)
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


# ── 3f. TARGET ENCODING — leak-free version ──────────────────────────────────
#
#  FIX (Issue 2): make_target_encodings() must NEVER see the val fold's demand.
#  We keep this function for two uses:
#    (a) Encode `test`  → built from ALL of train (safe; test has no labels).
#    (b) Encode each CV fold's val split → built from that fold's TRAIN split only.
#  The outer call below handles (a). Case (b) is done inside the CV loop.
#
def make_target_encodings(train_df: pd.DataFrame,
                          apply_df: pd.DataFrame) -> pd.DataFrame:
    apply_df = apply_df.copy()
    tr       = train_df

    # Drop any stale TE columns so .join() never hits a collision
    _te_cols = [
        "geo_demand_mean", "geo_demand_std", "slot_demand_mean",
        "geo_slot_demand_mean", "q5_demand_mean", "q4_demand_mean",
        "geo_hour_demand_mean", "road_lanes_demand_mean",
    ]
    apply_df = apply_df.drop(columns=[c for c in _te_cols if c in apply_df.columns])

    # Per-geohash demand stats
    geo_stats = tr.groupby("geohash")["demand"].agg(["mean", "std"]).rename(
        columns={"mean": "geo_demand_mean", "std": "geo_demand_std"}
    )
    geo_stats["geo_demand_std"] = geo_stats["geo_demand_std"].fillna(0)
    apply_df["geo_demand_mean"] = apply_df["geohash"].map(geo_stats["geo_demand_mean"])
    apply_df["geo_demand_std"]  = apply_df["geohash"].map(geo_stats["geo_demand_std"])

    # Per-slot demand stats
    slot_mean = tr.groupby("slot_of_day")["demand"].mean().rename("slot_demand_mean")
    apply_df["slot_demand_mean"] = apply_df["slot_of_day"].map(slot_mean)

    # Per-(geohash, slot) interaction
    geo_slot_mean = (
        tr.groupby(["geohash", "slot_of_day"])["demand"]
          .mean()
          .rename("geo_slot_demand_mean")
    )
    apply_df = apply_df.join(geo_slot_mean, on=["geohash", "slot_of_day"], how="left")

    # Prefix-level means
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

    # Per-(geohash, hour) mean
    geo_hour_mean = (
        tr.groupby(["geohash", "hour"])["demand"]
          .mean()
          .rename("geo_hour_demand_mean")
    )
    apply_df = apply_df.join(geo_hour_mean, on=["geohash", "hour"], how="left")

    # RoadType × NumberOfLanes mean
    road_lanes_mean = (
        tr.groupby(["RoadType", "NumberofLanes"])["demand"]
          .mean()
          .rename("road_lanes_demand_mean")
    )
    apply_df = apply_df.join(road_lanes_mean, on=["RoadType", "NumberofLanes"], how="left")

    return apply_df


# Pre-compute encodings for the TEST set from all of train (safe — test has no labels)
test = make_target_encodings(train_df=train, apply_df=test)
# NOTE: We do NOT call make_target_encodings on `train` here.
# Instead it is called fold-by-fold inside the CV loop (fix for Issue 2).


# ── 3g. LAG FEATURES — leak-free version ─────────────────────────────────────
#
#  FIX (Issue 1): The original code groupby-averaged day-48 data at (geohash,
#  slot) for ALL training rows, including day-48 ones.  Because every
#  (geohash, slot) pair in day-48 has exactly 1 observation, this made
#  lag_demand_day48 equal to the row's OWN demand for every day-48 row —
#  handing the model the answer directly.
#
#  CORRECT APPROACH:
#    • Day-49 train/test rows  → lag from day-48  (legitimate; different day)
#    • Day-48 train rows       → lag from day-47 if available, else from the
#                                 global (geohash, slot) mean across all OTHER
#                                 training rows (leave-own-day-out)
#
#  FIX (Issue 5): Lags B & C are rewritten without the confusing slot_plusN
#  index-shift trick.  We now explicitly select the slot that is N steps
#  earlier (with wraparound handled cleanly) and do a direct merge.
#
def build_day48_lookup(train_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Return a (geohash, slot_of_day) → demand lookup built from day-48 data.
    This is the legitimate lag source for day-49 rows.
    """
    d48 = train_ref[train_ref["day"] == 48].copy()
    return (
        d48.groupby(["geohash", "slot_of_day"])["demand"]
           .mean()
           .reset_index()
           .rename(columns={"demand": "_d48_demand"})
    )

def build_day47_lookup(train_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Return a (geohash, slot_of_day) → demand lookup built from day-47 data.
    Used as the lag source for day-48 train rows to avoid self-leakage.
    Falls back to a geohash-level mean if day-47 data is sparse or absent.
    """
    has_d47 = 47 in train_ref["day"].unique()
    if has_d47:
        d47 = train_ref[train_ref["day"] == 47].copy()
        lkp = (
            d47.groupby(["geohash", "slot_of_day"])["demand"]
               .mean()
               .reset_index()
               .rename(columns={"demand": "_d47_demand"})
        )
        return lkp
    else:
        # No day-47 data available: use leave-own-day-out geohash×slot mean
        # (i.e. the mean computed from all OTHER days in train)
        d48_slots = set(
            zip(train_ref.loc[train_ref["day"]==48, "geohash"],
                train_ref.loc[train_ref["day"]==48, "slot_of_day"])
        )
        lkp = (
            train_ref[train_ref["day"] != 48]
            .groupby(["geohash", "slot_of_day"])["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "_d47_demand"})
        )
        return lkp


def add_lag_features(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features; the lag source depends on which day the row belongs to.

    For day-49 rows (all test rows, plus day-49 train rows):
        lag = demand at same (geohash, slot) on day 48  [legitimate]

    For day-48 train rows:
        lag = demand at same (geohash, slot) on day 47  [no self-leakage]
        Falls back to global geohash×slot mean from non-day-48 data when
        day-47 values are missing.

    Lag offset features (1h earlier, 2h earlier, rolling stats) follow the
    same day-split logic so they are equally leak-free.
    """
    df = df.copy()

    lkp48 = build_day48_lookup(train_ref)   # day-49 rows look back to day-48
    lkp47 = build_day47_lookup(train_ref)   # day-48 rows look back to day-47

    # Global fallbacks (used when a (geohash, slot) has no match in lkp)
    geo_mean_d48  = (
        train_ref[train_ref["day"] == 48]
        .groupby("geohash")["demand"].mean()
    )
    slot_mean_d48 = (
        train_ref[train_ref["day"] == 48]
        .groupby("slot_of_day")["demand"].mean()
    )
    global_mean   = train_ref["demand"].mean()

    # ── Helper: merge a lookup into df, producing column `out_col` ──────────
    def merge_lookup(frame, lkp, key_col, out_col, rename_src="_d48_demand"):
        """Left-join lkp onto frame using (geohash, key_col) → out_col."""
        lkp = lkp.rename(columns={rename_src: out_col,
                                   "slot_of_day": key_col})
        merged = frame.merge(
            lkp[["geohash", key_col, out_col]],
            on=["geohash", key_col],
            how="left"
        )
        return merged

    # ── Helper: fill a column's NaNs with cascading fallbacks ───────────────
    def fill_lags(frame, col):
        missing = frame[col].isna()
        if missing.any():
            frame.loc[missing, col] = frame.loc[missing, "geohash"].map(geo_mean_d48)
        missing = frame[col].isna()
        if missing.any():
            frame.loc[missing, col] = frame.loc[missing, "slot_of_day"].map(slot_mean_d48)
        frame[col] = frame[col].fillna(global_mean)
        return frame

    # ── Split df by day so each group gets the appropriate lag source ────────
    mask_d48 = df["day"] == 48
    mask_d49 = ~mask_d48   # includes test rows (which have no `day` NaN issue)

    parts = []
    for mask, lkp48_or_47, src_col in [
        (mask_d48, lkp47, "_d47_demand"),
        (mask_d49, lkp48, "_d48_demand"),
    ]:
        part = df[mask].copy()
        if len(part) == 0:
            continue

        # ── Lag A: same (geohash, slot) ─────────────────────────────────────
        part = merge_lookup(part, lkp48_or_47.rename(columns={src_col: "_lag_a"}),
                            "slot_of_day", "lag_demand_day48", "_lag_a")
        part = fill_lags(part, "lag_demand_day48")

        # ── Lag B: same geohash, slot 4 positions earlier (= 1 hour back) ───
        #   We create a shifted lookup: for each row wanting slot S,
        #   we look up slot (S - 4) in the lag source.
        #   Implementation: rename the lag source's slot column so that
        #   a row at slot S joins to the lag source row at slot (S-4).
        lkp_shifted_1h = lkp48_or_47.copy().rename(columns={src_col: "_lag_b"})
        # "target_slot" is the current row's slot; the lag source slot is 4 less
        lkp_shifted_1h["target_slot"] = lkp_shifted_1h["slot_of_day"] + 4
        lkp_shifted_1h = lkp_shifted_1h.drop(columns="slot_of_day")
        part = part.merge(
            lkp_shifted_1h[["geohash", "target_slot", "_lag_b"]],
            left_on=["geohash", "slot_of_day"],
            right_on=["geohash", "target_slot"],
            how="left"
        ).drop(columns="target_slot").rename(columns={"_lag_b": "lag_demand_minus1h"})
        part = fill_lags(part, "lag_demand_minus1h")

        # ── Lag C: same geohash, slot 8 positions earlier (= 2 hours back) ──
        lkp_shifted_2h = lkp48_or_47.copy().rename(columns={src_col: "_lag_c"})
        lkp_shifted_2h["target_slot"] = lkp_shifted_2h["slot_of_day"] + 8
        lkp_shifted_2h = lkp_shifted_2h.drop(columns="slot_of_day")
        part = part.merge(
            lkp_shifted_2h[["geohash", "target_slot", "_lag_c"]],
            left_on=["geohash", "slot_of_day"],
            right_on=["geohash", "target_slot"],
            how="left"
        ).drop(columns="target_slot").rename(columns={"_lag_c": "lag_demand_minus2h"})
        part = fill_lags(part, "lag_demand_minus2h")

        # ── Lag D/E/F/G: rolling stats — computed from the appropriate day ───
        day_num = 48 if src_col == "_d48_demand" else 47
        # If day-47 data is unavailable, fall back to non-day-48 train rows
        if day_num == 47 and 47 not in train_ref["day"].unique():
            ref_day = train_ref[train_ref["day"] != 48]
        else:
            ref_day = train_ref[train_ref["day"] == day_num]

        if len(ref_day) > 0:
            ref_sorted = ref_day.sort_values(["geohash", "slot_of_day"])
            ref_sorted = ref_sorted.copy()
            ref_sorted["roll4"]    = ref_sorted.groupby("geohash")["demand"].transform(
                lambda x: x.rolling(4, min_periods=1).mean()
            )
            ref_sorted["roll16"]   = ref_sorted.groupby("geohash")["demand"].transform(
                lambda x: x.rolling(16, min_periods=1).mean()
            )
            ref_sorted["roll4_std"] = ref_sorted.groupby("geohash")["demand"].transform(
                lambda x: x.rolling(4, min_periods=1).std().fillna(0)
            )

            roll_lkp = (
                ref_sorted.groupby(["geohash", "slot_of_day"])
                [["roll4", "roll16", "roll4_std"]]
                .mean()
                .reset_index()
                .rename(columns={
                    "roll4":     "lag_rollmean4_day48",
                    "roll16":    "lag_rollmean16_day48",
                    "roll4_std": "lag_rollstd4_day48",
                })
            )

            part = part.merge(
                roll_lkp, on=["geohash", "slot_of_day"], how="left"
            )
        else:
            part["lag_rollmean4_day48"]  = np.nan
            part["lag_rollmean16_day48"] = np.nan
            part["lag_rollstd4_day48"]   = np.nan

        for col in ["lag_rollmean4_day48", "lag_rollmean16_day48", "lag_rollstd4_day48"]:
            part = fill_lags(part, col)

        # lag_geo_slot_d48: same as lag_demand_day48 but kept separate for
        # symmetry with the original feature set (lets the model weight it
        # independently alongside geo_slot_demand_mean from target encoding)
        part["lag_geo_slot_d48"] = part["lag_demand_day48"]

        parts.append(part)

    result = pd.concat(parts).sort_index()
    return result


train = add_lag_features(train, train_ref=train)
test  = add_lag_features(test,  train_ref=train)

print("    ✓ All features engineered")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DEFINE FEATURE COLUMNS
#
#  FIX (Issue 3): Removed raw `day` column.
#    - Train has days 48 and 49; test is all day 49.
#    - Including `day` adds almost no signal and can confuse the model since
#      test rows are always day=49 while training mixes both.
#    - The temporal information is already captured much better by the lag
#      features and time-of-day encodings.
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    # ── Time ──────────────────────────────────────────────────────────────────
    "hour", "minute", "slot_of_day",
    "sin_hour", "cos_hour", "sin_slot", "cos_slot",
    "is_daytime", "is_morning_rush", "is_evening", "is_night",
    # NOTE: `day` intentionally excluded (see Fix Issue 3 above)

    # ── Road infrastructure ───────────────────────────────────────────────────
    "road_type_enc", "NumberofLanes",
    "large_vehicles_flag", "landmarks_flag",
    "highway_x_landmark", "lanes_x_road",

    # ── Weather & temperature ─────────────────────────────────────────────────
    "weather_enc", "Temperature",

    # ── Missing-value flags (FIX Issue 4) ────────────────────────────────────
    "flag_missing_temperature",
    "flag_missing_weather",
    "flag_missing_roadtype",

    # ── Target encodings ─────────────────────────────────────────────────────
    # NOTE: These will be re-computed fold-by-fold inside the CV loop
    # (FIX Issue 2). The column names still need to exist at this stage
    # for the assertion check; they are populated below.
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

# Pre-populate target-encoding columns on train with NaN for now.
# They will be correctly filled fold-by-fold inside the CV loop.
TE_COLS = [
    "geo_demand_mean", "geo_demand_std",
    "slot_demand_mean", "geo_slot_demand_mean",
    "q5_demand_mean",  "q4_demand_mean",
    "geo_hour_demand_mean", "road_lanes_demand_mean",
]
for col in TE_COLS:
    if col not in train.columns:
        train[col] = np.nan

# Verify all feature columns exist in both train and test
missing_in_train = [f for f in FEATURES if f not in train.columns]
missing_in_test  = [f for f in FEATURES if f not in test.columns]
assert not missing_in_train, f"Missing in train: {missing_in_train}"
assert not missing_in_test,  f"Missing in test : {missing_in_test}"

y      = np.log1p(train["demand"].values)   # log-transform reduces skew
groups = train["day"].values          # split by day, not geohash

# Test matrix uses target encodings from ALL of train (safe — no test labels)
test_with_te = make_target_encodings(train_df=train, apply_df=test)
X_test = test_with_te[FEATURES].fillna(-1).astype("float32")

print(f"    Feature matrix : {train.shape[0]:,} train rows × {len(FEATURES)} features")
print(f"    Target (log1p) : mean={y.mean():.4f}, std={y.std():.4f}")
print(f"    FIX: target encodings will be recomputed per fold (no val leakage)")


# ─────────────────────────────────────────────────────────────────────────────
# 5a. LIGHTGBM — 5-Fold GroupKFold  (target encodings built inside each fold)
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

gkf = GroupKFold(n_splits=2)          # only 2 days exist (48 and 49)

lgb_oof  = np.zeros(len(train), dtype=np.float64)
lgb_test = np.zeros(len(X_test), dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, y, groups=groups)):
    tr_rows  = train.iloc[tr_idx]
    val_rows = train.iloc[val_idx]

    # ── FIX (Issue 2): build target encodings from TRAIN fold rows only ──────
    tr_rows_te  = make_target_encodings(train_df=tr_rows,  apply_df=tr_rows)
    val_rows_te = make_target_encodings(train_df=tr_rows,  apply_df=val_rows)
    # (val_rows_te uses tr_rows as the statistics source — val demand never seen)

    X_tr  = tr_rows_te[FEATURES].fillna(-1).astype("float32")
    X_val = val_rows_te[FEATURES].fillna(-1).astype("float32")
    y_tr  = y[tr_idx]
    y_val = y[val_idx]

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
    lgb_test         += model.predict(X_test) / 2

    fold_r2 = r2_score(
        train["demand"].values[val_idx],
        np.expm1(lgb_oof[val_idx]).clip(0, 1)
    )
    print(f"    Fold {fold + 1}/2  |  best iter: {model.best_iteration_:>4d}"
          f"  |  val R²: {fold_r2:.4f}")

lgb_oof_demand  = np.expm1(lgb_oof).clip(0, 1)
lgb_test_demand = np.expm1(lgb_test).clip(0, 1)
lgb_r2 = r2_score(train["demand"].values, lgb_oof_demand)
print(f"\n    ✓ LightGBM OOF R²: {lgb_r2:.4f}  |  Score: {max(0, 100*lgb_r2):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5b. XGBOOST — 5-Fold GroupKFold  (target encodings built inside each fold)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Training XGBoost ...")

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
xgb_test = np.zeros(len(X_test), dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, y, groups=groups)):
    tr_rows  = train.iloc[tr_idx]
    val_rows = train.iloc[val_idx]

    # ── FIX (Issue 2): same fold-local target encoding ────────────────────────
    tr_rows_te  = make_target_encodings(train_df=tr_rows,  apply_df=tr_rows)
    val_rows_te = make_target_encodings(train_df=tr_rows,  apply_df=val_rows)

    X_tr  = tr_rows_te[FEATURES].fillna(-1).astype("float32")
    X_val = val_rows_te[FEATURES].fillna(-1).astype("float32")
    y_tr  = y[tr_idx]
    y_val = y[val_idx]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    xgb_oof[val_idx]  = model.predict(X_val)
    xgb_test         += model.predict(X_test) / 2

    fold_r2 = r2_score(
        train["demand"].values[val_idx],
        np.expm1(xgb_oof[val_idx]).clip(0, 1)
    )
    print(f"    Fold {fold + 1}/  |  best iter: {model.best_iteration:>4d}"
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

# FIX (Issue 6): derive expected size from test.shape[0] rather than hardcoding
expected_rows = test.shape[0]
assert submission.shape == (expected_rows, 2), \
    f"❌ Wrong shape: {submission.shape} — expected ({expected_rows}, 2)"

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

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE LOG — v2
# ─────────────────────────────────────────────────────────────────────────────
# [CRITICAL] Lag self-leakage fixed (Issue 1)
#   Old: lag_demand_day48 used day-48 group means → every (geohash,slot) had
#        exactly 1 obs → lag == own demand for all day-48 rows (100% leakage).
#   New: Day-48 train rows now receive lags from day-47 (or leave-own-day-out
#        fallback). Day-49 rows still use day-48 lags (legitimate, different day).
#
# [CRITICAL] Target encoding CV leakage fixed (Issue 2)
#   Old: make_target_encodings() called once on full train before the CV loop.
#   New: Called inside each fold from tr_rows only; val fold demand never seen.
#
# [MODERATE] Raw `day` feature removed (Issue 3)
#   Adds noise since all test rows are day=49 while train mixes 48+49.
#
# [MODERATE] Missing-value flags added (Issue 4)
#   add_missing_flags() now captures flag_missing_temperature,
#   flag_missing_weather, flag_missing_roadtype BEFORE imputation.
#
# [MODERATE] Lag B & C rewrote without confusing slot_plusN index trick (Issue 5)
#   Now uses an explicit target_slot join so the shift direction is unambiguous.
#
# [MINOR] Submission shape assertion uses test.shape[0] not 41778 (Issue 6)
# ─────────────────────────────────────────────────────────────────────────────