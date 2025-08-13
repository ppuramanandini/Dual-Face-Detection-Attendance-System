import pandas as pd
import joblib

# -------- Step 1: Load trained model and tools --------
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler_s.pkl')
feature_columns = joblib.load('features_s.pkl')

# -------- Step 2: Create new input sample --------
new_data = {
    'RFID UID': [1567823],
    'Face Folder': ['sarika_555']
}

new_df = pd.DataFrame(new_data)

# -------- Step 3: One-hot encode and align features --------
new_df_encoded = pd.get_dummies(new_df)

# Add any missing columns (from training) with 0s
for col in feature_columns:
    if col not in new_df_encoded.columns:
        new_df_encoded[col] = 0

# Ensure correct column order
X_new = new_df_encoded[feature_columns]

# -------- Step 4: Scale and predict --------
X_new_scaled = scaler.transform(X_new)
prediction = model.predict(X_new_scaled)

# -------- Step 5: Output the prediction --------
print(f"🎯 Predicted Student ID: {prediction[0]:.2f}")