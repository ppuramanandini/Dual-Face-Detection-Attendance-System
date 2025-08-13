import cv2
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# -------- Configuration --------
student_name = "mounika_537"  # Change as needed
image_dir = "images"
save_dir = os.path.join(image_dir, student_name)
os.makedirs(save_dir, exist_ok=True)

# -------- Step 1: Capture 5 photos using SPACE key --------
cap = cv2.VideoCapture(0)
photo_count = 0

print("📸 Press SPACE to capture a photo. Press ESC to exit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Photo Capture - Press SPACE", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to quit
        print("❌ Capture cancelled by user.")
        break
    elif key == 32:  # SPACE bar to capture
        photo_path = os.path.join(save_dir, f"{student_name}_{photo_count+1}.jpg")
        cv2.imwrite(photo_path, frame)
        photo_count += 1
        print(f"✅ Captured photo {photo_count}/5")

        if photo_count >= 5:
            print("📁 5 photos captured. Closing camera.")
            break

cap.release()
cv2.destroyAllWindows()

# -------- Step 2: Train a sample model (dummy data) --------
# Sample dummy data (replace with real CSV if needed)
data = {
    'RFID UID': [1567823, 1567824, 1567825, 1567826, 1567827],
    'Face Folder': ['sai_008', 'john_001', 'amy_002', 'mike_004', 'sara_003'],
    'Student ID': [6, 7, 8, 9, 10]
}

df = pd.DataFrame(data)

# One-hot encode 'Face Folder'
df_encoded = pd.get_dummies(df, columns=['Face Folder'])

# Prepare features and target
X = df_encoded.drop('Student ID', axis=1)
y = df_encoded['Student ID']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model, scaler, and feature list
joblib.dump(model, 'student_model.pkl')
joblib.dump(scaler, 'scaler_s.pkl')
joblib.dump(X.columns.tolist(), 'features_s.pkl')

# -------- Final message --------
print("\n✅ 5 photos saved in folder:", save_dir)
print("✅ Model trained and saved successfully.")