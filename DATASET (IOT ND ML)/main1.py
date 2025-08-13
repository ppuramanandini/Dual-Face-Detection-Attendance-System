import cv2
import os
import csv

os.makedirs("dataset", exist_ok=True)

csv_file = "student_data.csv"
file_exists = os.path.isfile(csv_file)
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Student ID", "Name", "RFID UID", "Face Folder"])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

for i in range(10):  
    print(f"\n===== Student {i+1} Registration =====")
    student_id = input("Enter Student ID: ")
    name = input("Enter Student Name: ")
    rfid_uid = input("Enter (fake) RFID UID: ")

    folder_name = f"{name.lower()}_{student_id}"
    folder_path = os.path.join("dataset", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([student_id, name, rfid_uid, folder_name])


    print(f"\n[INFO] Capturing face images for {name}. Look at the camera...")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not working.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{folder_path}/{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces - Press 'q' to skip", frame)
        if cv2.waitKey(1) == ord('q') or count >= 50:
            break

    print(f"[INFO] {count} face images saved in {folder_path}")

cap.release()
cv2.destroyAllWindows()
print("\n[DONE] All 10 student datasets created successfully.")
