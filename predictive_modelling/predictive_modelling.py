import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


# Load and prepare data (updated for your specific CSV structure)
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)

    # Convert relevant columns to appropriate types
    data['Date Posted'] = pd.to_datetime(data['Date Posted'])
    data['Day Uploaded'] = pd.to_datetime(data['Day Uploaded'])

    # Create new features
    data['Hour Posted'] = data['Date Posted'].dt.hour
    data['Month Posted'] = data['Date Posted'].dt.month
    data['Year Posted'] = data['Date Posted'].dt.year
    data['Days Since Upload'] = (pd.to_datetime('today') - data['Day Uploaded']).dt.days
    data['Shawn Screen Ratio'] = data['Shawn Seconds'] / data['Total Seconds'].replace(0, np.nan)
    data['Shawn Screen Ratio'] = data['Shawn Screen Ratio'].fillna(0)

    # Calculate engagement features
    data['Engagement Rate'] = (data['Likes'] + data['Comments']) / data['Views'].replace(0, np.nan)
    data['Engagement Rate'] = data['Engagement Rate'].fillna(0)

    # Handle infinite values from division
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # Log transform target variable to handle skewness
    data['Log_Likes'] = np.log1p(data['Likes'])

    return data


# Load data
file_path = r'C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger\refined_combined_with_topics.csv'
df = load_and_preprocess(file_path)

# Define features and target (updated based on your columns)
categorical_features = ['Dominant Emotion', 'Assigned Topic', 'Weekday vs. Weekend Upload']
numerical_features = [
    'Screen Time Percentage',
    'Hour Posted',
    'Month Posted',
    'Year Posted',
    'Days Since Upload',
    'Shawn Screen Ratio',
    'Views per Day',
    'Engagement Rate',
    'Like-to-Comment Ratio'
]
target = 'Log_Likes'

X = df[numerical_features + categorical_features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# # Models to test
# models = {
#     'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
#     'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
#                                    solver='adam', early_stopping=True, random_state=42)
# }
#
# # Evaluation
# results = {}
# for name, model in models.items():
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('model', model)
#     ])
#
#     pipeline.fit(X_train, y_train)
#
#     # Predictions for both train and test
#     y_train_pred = pipeline.predict(X_train)
#     y_test_pred = pipeline.predict(X_test)
#
#     # Convert predictions back from log scale
#     y_train_exp = np.expm1(y_train)
#     y_train_pred_exp = np.expm1(y_train_pred)
#     y_test_exp = np.expm1(y_test)
#     y_test_pred_exp = np.expm1(y_test_pred)
#
#     # Calculate metrics
#     train_r2 = r2_score(y_train_exp, y_train_pred_exp)
#     train_mse = mean_squared_error(y_train_exp, y_train_pred_exp)
#     test_r2 = r2_score(y_test_exp, y_test_pred_exp)
#     test_mse = mean_squared_error(y_test_exp, y_test_pred_exp)
#
#     results[name] = {
#         'Train R2': train_r2,
#         'Train MSE': train_mse,
#         'Test R2': test_r2,
#         'Test MSE': test_mse,
#         'Model': pipeline
#     }
#
#     # Print results with comparison note
#     print(f"\n{name}:")
#     print(f"  Training R2: {train_r2:.4f}  |  Training MSE: {train_mse:.4f}")
#     print(f"  Test R2:     {test_r2:.4f}  |  Test MSE:     {test_mse:.4f}")
#
#     # Check for overfitting
#     r2_gap = train_r2 - test_r2
#     if r2_gap > 0.2:
#         print("  ⚠️ Large gap between train/test R2 - possible overfitting!")
#     elif r2_gap > 0.1:
#         print("  ⚠️ Moderate gap between train/test R2 - potential overfitting.")
#     print("-" * 60)

# Update the Neural Network configuration in the models dictionary
models = {
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'Neural Network': MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        early_stopping=True,
        max_iter=1000,  # Increased from default 200
        random_state=42,
        alpha=0.001,  # Added L2 regularization
        batch_size=32  # Added batch processing
    )
}

# Modified evaluation section with overflow protection
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    # Predictions with clipping to prevent overflow
    y_train_pred = np.clip(pipeline.predict(X_train), -700, 700)
    y_test_pred = np.clip(pipeline.predict(X_test), -700, 700)

    # Safe conversion from log scale with clipping
    y_train_exp = np.expm1(y_train)
    y_train_pred_exp = np.expm1(y_train_pred)
    y_test_exp = np.expm1(y_test)
    y_test_pred_exp = np.expm1(y_test_pred)

    # Calculate metrics
    train_r2 = r2_score(y_train_exp, y_train_pred_exp)
    train_mse = mean_squared_error(y_train_exp, y_train_pred_exp)
    test_r2 = r2_score(y_test_exp, y_test_pred_exp)
    test_mse = mean_squared_error(y_test_exp, y_test_pred_exp)

    results[name] = {
        'Train R2': train_r2,
        'Train MSE': train_mse,
        'Test R2': test_r2,
        'Test MSE': test_mse,
        'Model': pipeline
    }

    # Print results
    print(f"\n{name}:")
    print(f"  Training R2: {train_r2:.4f}  |  Training MSE: {train_mse:.4f}")
    print(f"  Test R2:     {test_r2:.4f}  |  Test MSE:     {test_mse:.4f}")

    # Overfitting check
    r2_gap = train_r2 - test_r2
    if r2_gap > 0.2:
        print("  ⚠️ Large gap between train/test R2 - possible overfitting!")
    elif r2_gap > 0.1:
        print("  ⚠️ Moderate gap between train/test R2 - potential overfitting.")
    print("-" * 60)

# Best model based on Test R2
best_model_name = max(results, key=lambda x: results[x]['Test R2'])
best_model = results[best_model_name]['Model']
print(f"\nBest model: {best_model_name} with Test R2 score: {results[best_model_name]['Test R2']:.4f}")

# Feature importance
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features = ohe.get_feature_names_out(categorical_features)
    all_features = numerical_features + list(cat_features)
    importances = best_model.named_steps['model'].feature_importances_
    fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(r'C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger\feature_importances.png')
    plt.show()

# Sample prediction
sample_idx = 0
sample_data = X_test.iloc[[sample_idx]]
sample_pred = np.expm1(best_model.predict(sample_data))
sample_actual = np.expm1(y_test.iloc[sample_idx])
print("\nSample Prediction:")
print(f"Predicted Likes: {sample_pred[0]:.1f}")
print(f"Actual Likes: {sample_actual:.1f}")
print("\nSample Data Features:")
print(sample_data)