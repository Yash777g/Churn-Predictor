## Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
import joblib

## Load dataset
try:
    df = pd.read_csv("Insert dataset location !!!!")
    print("Data loaded successfully!\nFirst 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("ERROR: File not found. Download from:")
    print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    exit()


## Data Cleaning

# Handle TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

## Feature Engineering

# Interaction feature: Customer value
df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']

## One-Hot Encode categorical variables

cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod']

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


## Split features and target

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

## Scale numerical features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'tenure_MonthlyCharges']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

## Handle imbalance with SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTEENN:")
print(pd.Series(y_train_res).value_counts())

## XGBoost Hyperparameter Tuning
xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = RandomizedSearchCV(
    xgb, param_distributions=param_grid, n_iter=20, scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1, random_state=42
)

grid.fit(X_train_res, y_train_res)

best_xgb = grid.best_estimator_
print("\nBest XGBoost Parameters:", grid.best_params_)

## Evaluate on Test Set
y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)[:,1]

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("Average Precision:", average_precision_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = best_xgb.feature_importances_
features = X.columns
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
plt.title("Top 10 Important Features (XGBoost)")
plt.tight_layout()
plt.show()

## Save Model and Scaler
joblib.dump(best_xgb, "best_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel saved as 'best_churn_model.pkl' and scaler as 'scaler.pkl'")

## Prediction Function
def predict_churn(customer_data):
    """Predict churn for new customer data"""
    # Load model and scaler
    model = joblib.load("best_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    customer_df = pd.DataFrame([customer_data])
    
    # Feature engineering
    customer_df['tenure_MonthlyCharges'] = customer_df['tenure'] * customer_df['MonthlyCharges']
    
    # One-hot encode categorical features (align with training columns)
    customer_df = pd.get_dummies(customer_df)
    missing_cols = set(X.columns) - set(customer_df.columns)
    for c in missing_cols:
        customer_df[c] = 0
    customer_df = customer_df[X.columns]  # reorder columns
    
    # Scale numerical features
    customer_df[num_cols] = scaler.transform(customer_df[num_cols])
    
    # Predict
    pred = model.predict(customer_df)[0]
    prob = model.predict_proba(customer_df)[0][1]
    
    return {'Will Churn': 'Yes' if pred==1 else 'No', 'Probability': f"{prob:.2%}"}

## Example Prediction
example_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.35,
    'TotalCharges': 850.75
}

print("\nExample Prediction:", predict_churn(example_customer))
