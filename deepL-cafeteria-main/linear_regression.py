import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('daily_sort.csv')

# Aggregate the data to get total counts for each product
product_counts = data.melt(id_vars=['date', 'DayOfWeek', 'total_count'], 
                           value_vars=['ProductName_1', 'ProductName_2', 'ProductName_3', 'ProductName_4'],
                           var_name='ProductType', value_name='ProductName')

product_counts['Count'] = data.melt(id_vars=['date', 'DayOfWeek', 'total_count'], 
                                    value_vars=['count_1', 'count_2', 'count_3', 'count_4'],
                                    var_name='CountType', value_name='Count')['Count']

# Normalize the counts by total_count
product_counts['NormalizedCount'] = product_counts['Count'] / product_counts['total_count']

# Calculate average popularity for each product
product_popularity = product_counts.groupby('ProductName')['NormalizedCount'].mean().reset_index()
print(product_popularity)
# Prepare data for model
# For simplicity, let's assume we are predicting the normalized count difference between two products
# Create a dataset where each row is a pair of products and the target is the difference in their normalized counts

# Example: Create all possible pairs of products
from itertools import combinations
product_pairs = list(combinations(product_popularity['ProductName'], 2))

pair_data = []
for pair in product_pairs:
    product1, product2 = pair
    pop1 = product_popularity[product_popularity['ProductName'] == product1]['NormalizedCount'].values[0]
    pop2 = product_popularity[product_popularity['ProductName'] == product2]['NormalizedCount'].values[0]
    pair_data.append({'Product1': product1, 'Product2': product2, 'PopularityDiff': pop1 - pop2})

pair_df = pd.DataFrame(pair_data)

# Split data into training and testing sets
X = pair_df[['Product1', 'Product2']]
y = pair_df['PopularityDiff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode product names as categorical variables
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Ensure both train and test have the same columns
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_encoded, y_train)

# Predict on the test set
y_pred = model.predict(X_test_encoded)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction
def predict_relative_demand(product1, product2):
    input_data = pd.DataFrame({'Product1': [product1], 'Product2': [product2]})
    input_encoded = pd.get_dummies(input_data).reindex(columns=X_train_encoded.columns, fill_value=0)
    diff = model.predict(input_encoded)[0]
    if diff > 0:
        return f"{product1} is more popular by {diff*100:.2f}%"
    else:
        return f"{product2} is more popular by {-diff*100:.2f}%"

# Test the function
print(predict_relative_demand('Japanese Katsu Curry', 'Palak Paneer'))