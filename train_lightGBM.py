import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import pickle
import os # Added for directory creation

# --- Configuration & Constants ---
CLEANED_DATA_FILE = 'Update_25_LunchData.pkl'
MODELS_OUTPUT_DIR = 'models_lgbm' # Main output directory for this model type

# Define specific file paths within the output directory
MODEL_SAVE_PATH = os.path.join(MODELS_OUTPUT_DIR, 'sales_predictor_lgbm.pkl')
SCALER_PIPELINE_SAVE_PATH = os.path.join(MODELS_OUTPUT_DIR, 'preprocessor_lgbm.pkl')
DISH_AVG_SALES_SAVE_PATH = os.path.join(MODELS_OUTPUT_DIR, 'dish_avg_sales_lgbm.pkl')
FEATURE_IMPORTANCE_T1_PATH = os.path.join(MODELS_OUTPUT_DIR, 'feature_importances_target_1_lgbm.png')
FEATURE_IMPORTANCE_T2_PATH = os.path.join(MODELS_OUTPUT_DIR, 'feature_importances_target_2_lgbm.png')
TARGET_DIST_PLOT_PATH = os.path.join(MODELS_OUTPUT_DIR, 'target_variable_distribution.png')


UNKNOWN_DISH_TOKEN = 'UNK_DISH'

LGBM_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'regression_l1',
    'metric': 'mae',
    # Consider adding some L1/L2 regularization to combat overfitting
    # 'reg_alpha': 0.1, # L1 regularization
    # 'reg_lambda': 0.1 # L2 regularization
}

# --- Create Output Directory ---
if not os.path.exists(MODELS_OUTPUT_DIR):
    os.makedirs(MODELS_OUTPUT_DIR)
    print(f"Created output directory: {MODELS_OUTPUT_DIR}")

# --- 1. Load Cleaned Data ---
print("--- 1. Loading Cleaned Data ---")
try:
    df = pd.read_pickle(CLEANED_DATA_FILE)
    print(f"Successfully loaded '{CLEANED_DATA_FILE}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: Cleaned data file '{CLEANED_DATA_FILE}' not found. Please run the cleanup script first.")
    exit()

# --- Display Target Variable Stats & Plot Distribution ---
print("\n--- Target Variable Statistics ---")
print("Count_1 stats:\n", df['count_1'].describe())
print("\nCount_2 stats:\n", df['count_2'].describe())
df[['count_1', 'count_2']].hist(bins=30, figsize=(10,4))
plt.suptitle("Distribution of count_1 and count_2")
plt.savefig(TARGET_DIST_PLOT_PATH)
print(f"Target variable distribution plot saved to {TARGET_DIST_PLOT_PATH}")
plt.close()


# --- 2. Feature Engineering ---
print("\n--- 2. Feature Engineering ---")
print("Calculating inherent dish popularity (average sales)...")
sales_data = []
for i in range(1, 5):
    prod_col, count_col = f'ProductName_{i}', f'count_{i}'
    if prod_col in df.columns and count_col in df.columns:
        subset = df[[prod_col, count_col]].copy()
        subset.rename(columns={prod_col: 'DishName', count_col: 'Count'}, inplace=True)
        sales_data.append(subset)
if not sales_data: exit("ERROR: No product/count columns for popularity calculation.")
all_sales_df = pd.concat(sales_data).dropna()
dish_avg_sales = all_sales_df.groupby('DishName')['Count'].mean().to_dict()
fallback_avg_sale = np.mean(list(dish_avg_sales.values())) if dish_avg_sales else 0
if UNKNOWN_DISH_TOKEN not in dish_avg_sales:
    dish_avg_sales[UNKNOWN_DISH_TOKEN] = fallback_avg_sale

df['AvgSales_P1'] = df['ProductName_1'].map(dish_avg_sales).fillna(dish_avg_sales.get(UNKNOWN_DISH_TOKEN, fallback_avg_sale))
df['AvgSales_P2'] = df['ProductName_2'].map(dish_avg_sales).fillna(dish_avg_sales.get(UNKNOWN_DISH_TOKEN, fallback_avg_sale))
df['PopularityDiff_P1P2'] = (df['AvgSales_P1'] - df['AvgSales_P2']).abs()
df['PopularityRatio_P1P2'] = (df['AvgSales_P1'] + 1e-6) / (df['AvgSales_P2'] + 1e-6)
df['Expected_Total_Count'] = df['total_count']
df['Is_Katsu_Friday_Rule'] = (
    ((df['ProductName_1'] == 'Japanese Katsu Curry') | (df['ProductName_2'] == 'Japanese Katsu Curry')) &
    (df['DayOfWeek'].astype(str).str.lower() == 'friday') # Ensure DayOfWeek is string for .str access
).astype(int)
print(f"Katsu rule applied. Instances: {df['Is_Katsu_Friday_Rule'].sum()}")


# --- 3. Define Features (X) and Target (y) ---
print("\n--- 3. Defining Features and Target ---")
categorical_features = ['ProductName_1', 'ProductName_2', 'DayOfWeek']
numerical_features = ['AvgSales_P1', 'AvgSales_P2', 'PopularityDiff_P1P2',
                      'PopularityRatio_P1P2', 'Expected_Total_Count', 'Is_Katsu_Friday_Rule']
# weather_features = ['temp_c', 'temp_min_c', 'temp_max_c', 'feelslike_c', 'humidity_pct', 'precip_mm']
# numerical_features.extend(weather_features) # Uncomment to include weather

X = df[categorical_features + numerical_features].copy() # Use .copy() to avoid SettingWithCopyWarning later
y = df[['count_1', 'count_2']].values

# --- 4. Preprocessing ---
print("\n--- 4. Preprocessing ---")
# Ensure categorical features are string type for OneHotEncoder
for col in categorical_features:
    X[col] = X[col].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.03), categorical_features), # min_frequency to group rare
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

# --- 5. Splitting Data ---
print("\n--- 5. Splitting Data ---")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print("Fitting and transforming data with preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

feature_names_out = None
try:
    feature_names_out = preprocessor.get_feature_names_out()
    print(f"Number of features after preprocessing: {len(feature_names_out)}")
except AttributeError:
    print("Could not get feature names from preprocessor (older scikit-learn or issue).")


# --- 6. Model Training ---
print("\n--- 6. Model Training (LightGBM with MultiOutputRegressor) ---")
lgbm_regressor = lgb.LGBMRegressor(**LGBM_PARAMS)
multioutput_model = MultiOutputRegressor(lgbm_regressor)
print("Training the multioutput LightGBM model...")
multioutput_model.fit(X_train_processed, y_train)
print("Model training completed.")

# --- 7. Model Evaluation ---
print("\n--- 7. Model Evaluation ---")
y_pred_train = multioutput_model.predict(X_train_processed)
y_pred_val = multioutput_model.predict(X_val_processed)
mae_train_t1 = mean_absolute_error(y_train[:, 0], y_pred_train[:, 0])
mae_train_t2 = mean_absolute_error(y_train[:, 1], y_pred_train[:, 1])
overall_mae_train = mean_absolute_error(y_train, y_pred_train)
mae_val_t1 = mean_absolute_error(y_val[:, 0], y_pred_val[:, 0])
mae_val_t2 = mean_absolute_error(y_val[:, 1], y_pred_val[:, 1])
overall_mae_val = mean_absolute_error(y_val, y_pred_val)

print(f"\nTraining MAE - Count 1: {mae_train_t1:.2f}")
print(f"Training MAE - Count 2: {mae_train_t2:.2f}")
print(f"Overall Training MAE: {overall_mae_train:.2f}")
print(f"\nValidation MAE - Count 1: {mae_val_t1:.2f}")
print(f"Validation MAE - Count 2: {mae_val_t2:.2f}")
print(f"Overall Validation MAE: {overall_mae_val:.2f}")


# --- 8. Feature Importances ---
print("\n--- 8. Feature Importances ---")
if feature_names_out is not None:
    for i, estimator in enumerate(multioutput_model.estimators_):
        target_name = f"Count {i+1}"
        print(f"\nFeature importances for {target_name}:")
        importances = estimator.feature_importances_
        
        # Create a DataFrame for better sorting and access to names
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_out,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        plt.figure(figsize=(12, 8)) # Adjusted size
        top_n = min(20, len(feature_importance_df)) # Plot top 20 or fewer
        plt.barh(feature_importance_df['feature'][:top_n][::-1], # Horizontal bar plot, reversed for top-down
                 feature_importance_df['importance'][:top_n][::-1])
        plt.xlabel("LightGBM Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Feature Importances for {target_name}")
        plt.tight_layout()
        
        save_path = os.path.join(MODELS_OUTPUT_DIR, f'feature_importances_{target_name.replace(" ", "_").lower()}_lgbm.png')
        plt.savefig(save_path)
        print(f"Feature importance plot saved for {target_name} as {save_path}")
        plt.close() # Close plot
else:
    print("Skipping feature importance plotting as feature names could not be retrieved.")


# --- 9. Save Model and Preprocessor ---
print("\n--- 9. Save Model and Preprocessor ---")
try:
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(multioutput_model, f)
    print(f"Trained LightGBM multioutput model saved to {MODEL_SAVE_PATH}")
    with open(SCALER_PIPELINE_SAVE_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor (scaler & OHE) saved to {SCALER_PIPELINE_SAVE_PATH}")
    with open(DISH_AVG_SALES_SAVE_PATH, 'wb') as f:
        pickle.dump(dish_avg_sales, f)
    print(f"dish_avg_sales mapping saved to {DISH_AVG_SALES_SAVE_PATH}")
except Exception as e:
    print(f"ERROR during saving: {e}")

print("\n--- Script Finished ---")