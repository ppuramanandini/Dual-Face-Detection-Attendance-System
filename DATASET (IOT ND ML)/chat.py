import face_recognition
import cv2
import os
import numpy as np

# Step 1: Load known face encodings
known_encodings = []
known_names = []

image_dir = "images"  # Each subfolder = 1 person
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
               
            else:
                print(f"⚠️ No face found in {image_path}")
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
        break  # Use only one image per person

if not known_encodings:
    print("❌ No known faces loaded. Exiting.")
    exit()

# Step 2: Start webcam
cap = cv2.VideoCapture(0)
print("\n🎥 Looking for face... (Press ESC to exit)")

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
            print("📌 You got the attendance!")
            attendance_given = True
        else:
            print(f"\n❌ Face not recognized (Distance: {best_distance:.2f})")

        break  # Handle only one face per frame

    if attendance_given:
        break

    cv2.imshow('Face Recognition - Press ESC to Exit', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

if not attendance_given:
    print("\n❌ No valid attendance recorded.")
