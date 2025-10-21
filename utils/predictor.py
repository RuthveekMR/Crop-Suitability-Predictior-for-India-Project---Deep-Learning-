# utils/predictor.py
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scipy.spatial.distance import cdist
import os
import streamlit as st # Import Streamlit for caching
import warnings

# --- Asset Loading (Cached) ---
@st.cache_resource # Use Streamlit's resource caching for heavy objects
def load_assets(model_dir="model"):
    """Loads all necessary files on initialization and caches them."""
    print("--- 🔁 Initializing and loading all assets... ---")
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO messages
    try:
        model = tf.keras.models.load_model(os.path.join(model_dir, 'best_final_model.keras'))
        final_scalers = joblib.load(os.path.join(model_dir, 'final_scalers.pkl'))
        grid_lookup = pd.read_csv(os.path.join(model_dir, 'grid_lookup.csv'))
        ecocrop_df = pd.read_csv(os.path.join(model_dir, 'Ecocrop_cleaned_final_v5.csv'))
        print("--- ✅ All assets loaded successfully! ---")
        return model, final_scalers, grid_lookup, ecocrop_df
    except FileNotFoundError as e:
        st.error(f"❌ FATAL ERROR: Could not find a required asset file: {e.filename}. The app cannot start.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred during asset loading: {e}")
        st.stop()

# Load assets globally using the cached function when the script runs
MODEL, SCALERS, GRID_LOOKUP, CROP_DF = load_assets()

# --- Crop Database Creation ---
def _create_crop_database(ecocrop_df):
    """Processes the raw ecocrop CSV into the crop feature database."""
    # (Code for creating CROP_DB - same as previous version)
    column_rename_map = {
        'EcoPortCode': 'crop_id', 'COMNAME': 'crop_name',
        'temp_median': 'c_temp_median', 'temp_p25': 'c_temp_p25', 'temp_p75': 'c_temp_p75',
        'rf_median': 'c_rf_median', 'rf_p25': 'c_rf_p25', 'rf_p75': 'c_rf_p75',
        'soil_light': 'c_soil_light', 'soil_medium': 'c_soil_medium', 'soil_heavy': 'c_soil_heavy'
    }
    display_cols = ['ScientificName', 'LIFO', 'HABI', 'LISPA', 'PHYS', 'CAT', 'PHOTO', 'GMIN', 'GMAX']
    model_input_keys = list(column_rename_map.keys())
    all_needed_cols = list(set(['EcoPortCode', 'COMNAME'] + model_input_keys + display_cols))
    all_needed_cols = [col for col in all_needed_cols if col in ecocrop_df.columns]

    crop_db = ecocrop_df[all_needed_cols].rename(columns=column_rename_map)
    crop_db['crop_name'] = crop_db['crop_name'].fillna('Unknown Crop')
    for col in ['c_soil_light', 'c_soil_medium', 'c_soil_heavy']:
        if col in crop_db.columns:
            crop_db[col] = pd.to_numeric(crop_db[col], errors='coerce').fillna(0).astype(int)
    for col in ['GMIN', 'GMAX']:
        if col in crop_db.columns:
            crop_db[col] = pd.to_numeric(crop_db[col], errors='coerce').fillna(np.nan)
    return crop_db.drop_duplicates(subset=['crop_id'])

CROP_DB = _create_crop_database(CROP_DF)

# --- Preprocessing Functions ---
# (_normalize_lat_lon, _apply_scalers, _add_engineered_features - same as previous version)
def _normalize_lat_lon(df):
    lat_min, lat_max, lon_min, lon_max = 6.0, 38.0, 68.0, 98.0
    df_out = df.copy()
    if 'g_lat' in df_out.columns:
        df_out['g_lat_s'] = (df_out['g_lat'] - lat_min) / (lat_max - lat_min)
    if 'g_lon' in df_out.columns:
        df_out['g_lon_s'] = (df_out['g_lon'] - lon_min) / (lon_max - lon_min)
    return df_out

def _apply_scalers(df):
    df_scaled = df.copy()
    feature_groups = {
        'grid_climate': ['g_temp_median', 'g_temp_p25', 'g_temp_p75', 'g_rf_median', 'g_rf_p25', 'g_rf_p75'],
        'crop_climate': ['c_temp_median', 'c_temp_p25', 'c_temp_p75', 'c_rf_median', 'c_rf_p25', 'c_rf_p75']
    }
    for group, cols in feature_groups.items():
        if group not in SCALERS: continue
        cols_in_df = [c for c in cols if c in df_scaled.columns]
        if not cols_in_df: continue
        df_scaled[cols_in_df] = df_scaled[cols_in_df].apply(pd.to_numeric, errors='coerce').fillna(0)
        try:
            robust_scaler, minmax_scaler = SCALERS[group]['robust'], SCALERS[group]['minmax']
            robust_transformed = robust_scaler.transform(df_scaled[cols_in_df])
            minmax_transformed = minmax_scaler.transform(robust_transformed)
            new_cols = [f"{c}_s" for c in cols_in_df]
            df_scaled[new_cols] = minmax_transformed
        except Exception as e:
            print(f"Warning: Scaling error for group {group}. {e}")
            new_cols = [f"{c}_s" for c in cols_in_df]
            df_scaled[new_cols] = 0.0 # Assign default value on error
    return df_scaled

def _add_engineered_features(df):
    df_eng = df.copy()
    required_cols_eng = {
        'g_temp_range_s': ['g_temp_p75_s', 'g_temp_p25_s'],
        'g_rainfall_variability_s': ['g_rf_p75_s', 'g_rf_p25_s'],
        'g_rainfall_to_temp_ratio_s': ['g_rf_median_s', 'g_temp_median_s'],
        'c_temp_range_s': ['c_temp_p75_s', 'c_temp_p25_s'],
        'c_rainfall_variability_s': ['c_rf_p75_s', 'c_rf_p25_s'],
        'c_rainfall_to_temp_ratio_s': ['c_rf_median_s', 'c_temp_median_s']
    }
    for new_col, src_cols in required_cols_eng.items():
        if all(c in df_eng.columns for c in src_cols):
            if 'range' in new_col or 'variability' in new_col:
                df_eng[new_col] = df_eng[src_cols[0]] - df_eng[src_cols[1]]
            elif 'ratio' in new_col:
                df_eng[new_col] = df_eng[src_cols[0]] / (df_eng[src_cols[1]] + 1e-6)
        else:
            df_eng[new_col] = 0.0
    return df_eng


def _get_grid_features(lat, lon, max_dist=1.5):
    """Finds nearest grid point and checks distance."""
    if 'g_lat' not in GRID_LOOKUP.columns or 'g_lon' not in GRID_LOOKUP.columns: return None
    distances = cdist(GRID_LOOKUP[['g_lat', 'g_lon']].values, np.array([[lat, lon]]))
    min_dist_idx, min_dist = np.argmin(distances), np.min(distances)
    if min_dist > max_dist: return None
    return GRID_LOOKUP.iloc[min_dist_idx].to_dict()

# --- Main Prediction Function (Exported for app.py) ---
# Ensure this function is defined at the TOP LEVEL (not indented inside anything)
def predict(lat, lon):
    """Main prediction pipeline for the Streamlit app."""
    # 1. Validate coordinates
    if not (6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0):
        return "out_of_bounds", None, None

    # 2. Get grid features (includes ocean check)
    grid_features = _get_grid_features(lat, lon)
    if grid_features is None:
        return "ocean_point", None, None

    # 3. Create all pairs
    grid_df = pd.DataFrame([grid_features])
    grid_df['key'], CROP_DB['key'] = 1, 1
    all_pairs = pd.merge(grid_df, CROP_DB.copy(), on='key', how='inner').drop('key', axis=1)
    if all_pairs.empty: return "success", grid_features, pd.DataFrame()

    # 4. Preprocess all pairs
    processed = _normalize_lat_lon(all_pairs)
    processed = _apply_scalers(processed)
    processed = _add_engineered_features(processed)

    # 5. Predict using the model
    MODEL_FEATURE_COLUMNS = [ # The exact 26 features your ResNet expects
        'g_lat_s', 'g_lon_s', 'g_temp_median_s', 'g_temp_p25_s', 'g_temp_p75_s',
        'g_rf_median_s', 'g_rf_p25_s', 'g_rf_p75_s', 'c_temp_median_s',
        'c_temp_p25_s', 'c_temp_p75_s', 'c_rf_median_s', 'c_rf_p25_s',
        'c_rf_p75_s', 'g_soil_light', 'g_soil_medium', 'g_soil_heavy',
        'c_soil_light', 'c_soil_medium', 'c_soil_heavy', 'g_temp_range_s',
        'g_rainfall_variability_s', 'g_rainfall_to_temp_ratio_s', 'c_temp_range_s',
        'c_rainfall_variability_s', 'c_rainfall_to_temp_ratio_s'
    ]
    X_pred = processed.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0.0)

    try:
        predictions = MODEL.predict(X_pred, verbose=0).flatten()
    except Exception as e: return "prediction_error", grid_features, None

    # 6. Rank and return results
    if 'crop_id' not in processed.columns: return "data_error", grid_features, None
    processed['suitability_score'] = predictions
    top_5 = processed.sort_values(by='suitability_score', ascending=False).head(5)

    return "success", grid_features, top_5
