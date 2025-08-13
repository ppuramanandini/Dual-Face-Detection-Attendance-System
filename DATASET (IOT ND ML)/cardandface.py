import face_recognition
import cv2
import os
import numpy as np

# ----------------- Simulate RFID Scan -----------------
def scan_card():
    print("🔐 Please scan your RFID card...")
    card = input("📥 Enter your card UID: ").strip()
    if card:
        print(f"✅ Card {card} scanned successfully.\n")
        return card
    else:
        print("❌ Invalid card.")
        return None

# ----------------- Load Known Faces -------------------
known_encodings = []
known_names = []

image_dir = "images"  # Folder: images/person_name/image.jpg
print("[INFO] Loading known faces from:", image_dir)

for person_name in os.listdir(image_dir):
    person_folder = os.path.join(image_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        image_path = os.path.join(person_folder, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"✅ Encoded: {person_name}")
            else:
                print(f"⚠️ No face found in {image_path}")
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
        break  # Only one image per person

if not known_encodings:
    print("❌ No known faces found. Exiting.")
    exit()

# ----------------- Step 1: RFID Scan ------------------
card_id = scan_card()
if not card_id:
    print("❌ Attendance denied. No card scanned.")
    exit()

# ----------------- Step 2: Face Recognition ------------
print("🎥 Starting face recognition... (Press ESC to exit)")

cap = cv2.VideoCapture(0)
tolerance = 0.45
attendance_given = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if not face_distances.any():
            continue

        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        if best_distance < tolerance:
            name = known_names[best_match_index]
            print(f"\n✅ Face matched: {name}")
            print(f"📌 Attendance granted for card {card_id} and face {name}.")
            print("🎉 You got the attendance!")
            attendance_given = True
        else:
            print(f"\n❌ Face not recognized (Distance: {best_distance:.2f})")

        break  # Only process one face per frame

    if attendance_given:
        break

    cv2.imshow('Face Recognition - Press ESC to Exit', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import face_recognition
import cv2
import numpy as np
import os
import time

# ---------------- RFID Setup ----------------
reader = SimpleMFRC522()
print("🔐 Waiting for RFID card...")

try:
    id, text = reader.read()
    card_id = str(id).strip()
    print(f"✅ RFID Card Scanned: {card_id}")

except Exception as e:
    print(f"❌ RFID read error: {e}")
    GPIO.cleanup()
    exit()

# ---------------- Load Known Faces ----------------
known_encodings = []
known_names = []
image_dir = "images"  # Folder: images/Name/image.jpg

print("[INFO] Loading known faces from:", image_dir)

for person_name in os.listdir(image_dir):
    person_folder = os.path.join(image_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        image_path = os.path.join(person_folder, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"✅ Encoded: {person_name}")
            else:
                print(f"⚠️ No face found in {image_path}")
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
        break  # Only one image per person

if not known_encodings:
    print("❌ No known faces found. Exiting.")
    exit()

# ---------------- Face Recognition ----------------
print("🎥 Starting webcam for face recognition... (ESC to exit)")

cap = cv2.VideoCapture(0)
tolerance = 0.45
attendance_given = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        if best_distance < tolerance:
            name = known_names[best_match_index]
            print(f"\n✅ Face matched: {name}")
            print(f"🎉 Attendance Granted: RFID = {card_id}, Face = {name}")
            attendance_given = True
        else:
            print(f"\n❌ Face not recognized (distance: {best_distance:.2f})")

        break  # Only check one face

    if attendance_given:
        break

    cv2.imshow('Face Recognition - Press ESC to Exit', frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()

if not attendance_given:
    print("\n❌ Attendance failed: Face not matched.")

if not attendance_given:
    print("\n❌ Attendance failed. Face not matched.")
