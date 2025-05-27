import pandas as pd
import numpy as np
import pickle
import os
from itertools import combinations # To generate pairs of dishes

# --- Configuration & Constants ---
MODELS_LGBM_DIR = 'models_lgbm' # Directory where LGBM artifacts are saved
MODEL_PATH = os.path.join(MODELS_LGBM_DIR, 'sales_predictor_lgbm.pkl')
PREPROCESSOR_PATH = os.path.join(MODELS_LGBM_DIR, 'preprocessor_lgbm.pkl')
DISH_AVG_SALES_PATH = os.path.join(MODELS_LGBM_DIR, 'dish_avg_sales_lgbm.pkl')
# We need a list of all known dishes to generate pairs.
# This could come from idx_to_dish.pkl if you saved it from the NN script,
# or we can derive it from dish_avg_sales keys.
# Alternatively, you might have a master list of all possible dishes.

UNKNOWN_DISH_TOKEN = 'UNK_DISH' # Must match what was used in training

# Prediction Parameters (User Defined)
PREDICTION_DAY_OF_WEEK = 'Wednesday' # Example: 'Monday', 'Tuesday', etc.
# Expected total sales for the prediction day. This is a crucial input.
# You might get this from historical averages, a separate forecast, or business estimates.
EXPECTED_TOTAL_DAILY_SALES = 400 # Example value
TOP_K_COMBINATIONS = 10 # How many top combinations to show

# Criteria for "Even Sales"
# Option 1: Absolute difference threshold
# EVEN_SALES_ABS_DIFF_THRESHOLD = 10 # e.g., difference is less than 10 units
# Option 2: Percentage difference threshold (relative to their average or total)
EVEN_SALES_PCT_DIFF_THRESHOLD = 0.20 # e.g., difference is less than 20% of their average sales

def load_artifacts():
    """Loads the trained model, preprocessor, and dish average sales map."""
    print("--- Loading Model and Artifacts ---")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")

        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")

        with open(DISH_AVG_SALES_PATH, 'rb') as f:
            dish_avg_sales_map = pickle.load(f)
        print(f"Dish average sales map loaded from {DISH_AVG_SALES_PATH}")

        # Get a list of known dishes (excluding UNK_DISH if present in keys)
        all_known_dishes = [dish for dish in dish_avg_sales_map.keys() if dish != UNKNOWN_DISH_TOKEN]
        if not all_known_dishes:
            # Fallback: try to load from an idx_to_dish if it exists (e.g. from NN training)
            # This part is optional and depends on what you have saved.
            # For now, we primarily rely on dish_avg_sales_map.
            print("WARNING: No dishes found in dish_avg_sales_map (excluding UNK). Pair generation might be limited.")


        return model, preprocessor, dish_avg_sales_map, all_known_dishes

    except FileNotFoundError as e:
        print(f"ERROR: Could not load artifact - {e}. Ensure training script was run and paths are correct.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None, None, None, None

def prepare_input_features(dish1_name, dish2_name, day_of_week, expected_total_sales, dish_avg_sales_map):
    """Prepares a single row DataFrame for prediction."""
    avg_sales_p1 = dish_avg_sales_map.get(dish1_name, dish_avg_sales_map.get(UNKNOWN_DISH_TOKEN, 0))
    avg_sales_p2 = dish_avg_sales_map.get(dish2_name, dish_avg_sales_map.get(UNKNOWN_DISH_TOKEN, 0))

    popularity_diff = abs(avg_sales_p1 - avg_sales_p2)
    popularity_ratio = (avg_sales_p1 + 1e-6) / (avg_sales_p2 + 1e-6)

    # Katsu rule feature generation
    katsu_condition = (
        ((dish1_name == 'Japanese Katsu Curry') or (dish2_name == 'Japanese Katsu Curry')) and
        (day_of_week.lower() == 'friday')
    )
    is_katsu_friday = int(katsu_condition) # Convert boolean to int (True -> 1, False -> 0)

    data = {
        'ProductName_1': [dish1_name],
        'ProductName_2': [dish2_name],
        'DayOfWeek': [day_of_week],
        'AvgSales_P1': [avg_sales_p1],
        'AvgSales_P2': [avg_sales_p2],
        'PopularityDiff_P1P2': [popularity_diff],
        'PopularityRatio_P1P2': [popularity_ratio],
        'Expected_Total_Count': [expected_total_sales],
        'Is_Katsu_Friday_Rule': [is_katsu_friday]
        # Add weather features here if your model was trained with them,
        # you'd need to provide their expected values for the prediction day.
        # 'temp_c': [DEFAULT_TEMP], ...
    }
    feature_df = pd.DataFrame(data)
    
    ordered_cols = ['ProductName_1', 'ProductName_2', 'DayOfWeek', # Categorical
                    'AvgSales_P1', 'AvgSales_P2', 'PopularityDiff_P1P2', # Numerical
                    'PopularityRatio_P1P2', 'Expected_Total_Count', 'Is_Katsu_Friday_Rule']
    
    # Add any missing weather columns with default values if they were part of training
    # default_weather_features = {
    #    'temp_c': 15, 'temp_min_c': 10, 'temp_max_c': 20, 
    #    'feelslike_c': 14, 'humidity_pct': 60, 'precip_mm': 0
    # } # Example defaults
    # weather_cols_in_training = [] # Define this list based on your training
    # for wf in weather_cols_in_training: 
    #    if wf not in feature_df.columns:
    #        feature_df[wf] = default_weather_features.get(wf, 0) # Get from defaults or use 0
    #    if wf not in ordered_cols: # Ensure it's added if it wasn't a primary feature
    #        ordered_cols.append(wf)


    # Ensure the DataFrame has all columns expected by the preprocessor, in the correct order.
    # If any are missing from 'ordered_cols' that the preprocessor was trained on,
    # they need to be added (e.g., with default values if they are numerical).
    # This step is important if `ordered_cols` isn't exhaustive of what preprocessor saw.
    # However, since `preprocessor.transform` expects the same columns as `fit_transform`,
    # ensuring `feature_df` contains exactly these `ordered_cols` before passing to transform is key.
    
    final_feature_df = pd.DataFrame(columns=ordered_cols) # Create df with correct column order
    for col in ordered_cols:
        if col in feature_df:
            final_feature_df[col] = feature_df[col]
        else:
            # This case should ideally not happen if ordered_cols and data dict are complete
            # For categorical, it would be UNKNOWN, for numerical a default (e.g. 0 or mean)
            # However, the OneHotEncoder in the preprocessor handles unknown categorical.
            # For numerical, they must be present.
            print(f"Warning: Column '{col}' was in ordered_cols but not in created data dict. This might be an issue.")
            # If it's a numerical feature that should exist, you might add a default:
            # if col in ['temp_c', ...]: final_feature_df[col] = [0] # some default
            # else: raise ValueError(f"Missing critical column: {col}")


    return final_feature_df[ordered_cols] # Return with columns in the specified order


def predict_sales_for_pair(model, preprocessor, input_df):
    """Makes a sales prediction for the input DataFrame."""
    try:
        processed_input = preprocessor.transform(input_df)
        # Predictions will be [pred_count_1, pred_count_2]
        predictions = model.predict(processed_input)
        # Ensure predictions are non-negative
        predictions = np.maximum(0, predictions)
        return predictions[0] # Return the first (and only) row of predictions
    except Exception as e:
        print(f"Error during prediction for input:\n{input_df}\nError: {e}")
        return np.array([0, 0]) # Return default on error


def check_even_sales(count1, count2, criteria_type="percentage", threshold=EVEN_SALES_PCT_DIFF_THRESHOLD):
    """Checks if sales are even based on the chosen criteria."""
    if count1 + count2 == 0: # Avoid division by zero if both are zero
        return True # Or False, depending on how you want to treat zero sales

    if criteria_type == "percentage":
        avg_sales = (count1 + count2) / 2
        if avg_sales == 0: return True # If average is zero, they are "even" at zero
        pct_diff = abs(count1 - count2) / avg_sales
        return pct_diff <= threshold
    elif criteria_type == "absolute":
        abs_diff = abs(count1 - count2)
        return abs_diff <= threshold
    else:
        raise ValueError("Invalid criteria_type. Choose 'percentage' or 'absolute'.")

def calculate_evenness_score(count1, count2):
    """Calculates a score where lower is more even. Can be used for ranking."""
    if count1 + count2 == 0:
        return float('inf') # Or 0 if you prefer to rank pairs with zero sales differently
    
    # Using normalized absolute difference (percentage of their sum)
    # Smaller score means more even.
    score = abs(count1 - count2) / (count1 + count2 + 1e-6) # +1e-6 to avoid div by zero
    return score


def main():
    """Main prediction and ranking script."""
    model, preprocessor, dish_avg_sales_map, all_known_dishes = load_artifacts()

    if not all_known_dishes or model is None:
        print("Exiting due to loading errors or no dishes to process.")
        return

    print(f"\n--- Generating Top {TOP_K_COMBINATIONS} Even Sales Combinations ---")
    print(f"For Day: {PREDICTION_DAY_OF_WEEK}, Expected Total Sales: {EXPECTED_TOTAL_DAILY_SALES}")
    print(f"Evenness Criteria: Percentage difference <= {EVEN_SALES_PCT_DIFF_THRESHOLD*100:.0f}%")

    # Generate all unique pairs of dishes
    # Ensure we don't pair a dish with itself or have duplicate pairs like (A,B) and (B,A)
    dish_pairs = list(combinations(all_known_dishes, 2))
    print(f"Generated {len(dish_pairs)} unique dish pairs to evaluate.")

    results = []
    processed_pairs = 0
    for dish1, dish2 in dish_pairs:
        input_df = prepare_input_features(dish1, dish2, PREDICTION_DAY_OF_WEEK,
                                          EXPECTED_TOTAL_DAILY_SALES, dish_avg_sales_map)
        
        pred_count1, pred_count2 = predict_sales_for_pair(model, preprocessor, input_df)
        
        # Ensure predicted counts are reasonable (e.g., not massively exceeding total expected)
        # This is a soft check; the model should ideally learn this.
        pred_count1 = min(pred_count1, EXPECTED_TOTAL_DAILY_SALES)
        pred_count2 = min(pred_count2, EXPECTED_TOTAL_DAILY_SALES)

        # if check_even_sales(pred_count1, pred_count2, threshold=EVEN_SALES_PCT_DIFF_THRESHOLD):
        #    evenness_score = calculate_evenness_score(pred_count1, pred_count2)
        #    results.append({
        #        'Dish1': dish1,
        #        'Dish2': dish2,
        #        'Pred_Count1': round(pred_count1),
        #        'Pred_Count2': round(pred_count2),
        #        'Total_Pair_Sales': round(pred_count1 + pred_count2),
        #        'Evenness_Score': evenness_score # Lower is better
        #    })
        
        # Store all results and filter/rank later, or filter now
        evenness_score = calculate_evenness_score(pred_count1, pred_count2)
        if evenness_score <= EVEN_SALES_PCT_DIFF_THRESHOLD : # Pre-filter based on threshold
             results.append({
               'Dish1': dish1,
               'Dish2': dish2,
               'Pred_Count1': round(pred_count1),
               'Pred_Count2': round(pred_count2),
               'Total_Pair_Sales': round(pred_count1 + pred_count2),
               'Evenness_Score': evenness_score # Lower is better
           })

        processed_pairs += 1
        if processed_pairs % 1000 == 0:
            print(f"Processed {processed_pairs}/{len(dish_pairs)} pairs...")

    if not results:
        print("No dish combinations met the even sales criteria.")
        return

    # Sort results by evenness_score (ascending, lower is better),
    # then perhaps by total pair sales (descending) as a tie-breaker
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['Evenness_Score', 'Total_Pair_Sales'], ascending=[True, False])

    print(f"\n--- Top {min(TOP_K_COMBINATIONS, len(results_df))} Recommended Dish Combinations ---")
    print(results_df.head(TOP_K_COMBINATIONS).to_string())

    # Optional: Save results to CSV
    output_csv_path = os.path.join(MODELS_LGBM_DIR, f'top_even_sales_combinations_{PREDICTION_DAY_OF_WEEK.lower()}.csv')
    results_df.head(TOP_K_COMBINATIONS).to_csv(output_csv_path, index=False)
    print(f"\nTop combinations saved to: {output_csv_path}")


if __name__ == '__main__':
    # --- Adjust these parameters before running ---
    PREDICTION_DAY_OF_WEEK = 'Friday' # <--- SET YOUR DESIRED DAY
    EXPECTED_TOTAL_DAILY_SALES = 350    # <--- SET YOUR EXPECTED TOTAL
    TOP_K_COMBINATIONS = 15
    EVEN_SALES_PCT_DIFF_THRESHOLD = 0.25 # Difference is less than 25% of their average

    main()