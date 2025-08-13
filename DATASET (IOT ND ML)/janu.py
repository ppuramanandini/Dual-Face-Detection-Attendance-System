from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import serial

# Serial communication setup
ard = serial.Serial(port='COM4', baudrate=115200, timeout=1)
time.sleep(2)

# Load known face encodings
print("[INFO] loading encodings + face detector...")
encodingsP = "encodings.pickle"
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize video stream
vs = VideoStream(src=0, framerate=10).start()
time.sleep(2.0)
fps = FPS().start()

# Track attendance to avoid duplicate signals
attendance_sent = {
    "sarika_555"
    ""
    
    
    ""
    ""
    ""
    ""
    "": False,
    "sowmya_556": False
}

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Detect faces
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            # Send signal only once per person
            if name == "sarika_555" and not attendance_sent["sarika_555"]:
                print("Recognized:", name)
                ard.write('A'.encode())
                attendance_sent["sarika_555"] = True

            elif name == "sowmya_556" and not attendance_sent["sowmya_556"]:
                print("Recognized:", name)
                ard.write('B'.encode())
                attendance_sent["sowmya_556"] = True

        names.append(name)

    # Draw face boxes and names
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

# Cleanup
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
ard.close()