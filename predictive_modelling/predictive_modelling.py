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


# Load and prepare data
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)

    # Select features and target
    features = ['Date Posted', 'Shawn Seconds', 'Total Seconds', 'Dominant Emotion',
                'Day Uploaded', 'Screen Time Percentage', 'Weekday vs. Weekend Upload', 'Topic_Label']
    target = 'Likes'
    df = data[features + [target]].copy()

    # Convert dates
    df['Date Posted'] = pd.to_datetime(df['Date Posted'])
    df['Day Uploaded'] = pd.to_datetime(df['Day Uploaded'])

    # Feature engineering
    df['Hour Posted'] = df['Date Posted'].dt.hour
    df['Month Posted'] = df['Date Posted'].dt.month
    df['Year Posted'] = df['Date Posted'].dt.year
    df['Days Since Upload'] = (pd.to_datetime('today') - df['Day Uploaded']).dt.days
    df['Shawn Screen Ratio'] = df['Shawn Seconds'] / df['Total Seconds'].replace(0, np.nan)
    df['Shawn Screen Ratio'] = df['Shawn Screen Ratio'].fillna(0)

    # Drop original columns
    df = df.drop(['Date Posted', 'Day Uploaded', 'Shawn Seconds', 'Total Seconds'], axis=1)

    # Log-transform target (helps with skewed like counts)
    df['Log_Likes'] = np.log1p(df[target])

    return df


# Load data
file_path = r'C:\Users\shann\PycharmProjects\capstone2025V2\outputs\topic_modeling_results.csv'
df = load_and_preprocess(file_path)

# Define features and target
categorical_features = ['Dominant Emotion', 'Topic_Label']
numerical_features = ['Screen Time Percentage', 'Weekday vs. Weekend Upload',
                      'Hour Posted', 'Month Posted', 'Year Posted',
                      'Days Since Upload', 'Shawn Screen Ratio']
target = 'Log_Likes'  # Using log-transformed target

X = df[numerical_features + categorical_features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
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

# Models to test
models = {
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                                   solver='adam', early_stopping=True, random_state=42)
}

# Evaluation
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Convert predictions back from log scale
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred)

    r2 = r2_score(y_test_exp, y_pred_exp)
    mse = mean_squared_error(y_test_exp, y_pred_exp)

    results[name] = {'R2 Score': r2, 'MSE': mse, 'Model': pipeline}

    print(f"{name}:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print("-" * 40)

# Best model
best_model_name = max(results, key=lambda x: results[x]['R2 Score'])
best_model = results[best_model_name]['Model']
print(f"\nBest model: {best_model_name} with R2 score: {results[best_model_name]['R2 Score']:.4f}")

# Feature importance (for tree-based models)
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    # Get feature names
    ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features = ohe.get_feature_names_out(categorical_features)
    all_features = numerical_features + list(cat_features)

    importances = best_model.named_steps['model'].feature_importances_
    fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.show()

# Sample prediction
sample_idx = 0
sample_data = X_test.iloc[[sample_idx]]
sample_pred = np.expm1(best_model.predict(sample_data))
sample_actual = np.expm1(y_test.iloc[sample_idx])

print("\nSample Prediction:")
print(f"Predicted Likes: {sample_pred[0]:.1f}")
print(f"Actual Likes: {sample_actual:.1f}")