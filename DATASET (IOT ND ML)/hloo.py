from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

# Load known face encodings
encodingsP = "encodings.pickle"
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# To store names that are already recognized
recognized_names = set()

# To store unknown face encodings already seen
unknown_encodings = []

# Start webcam
vs = VideoStream(src=0, framerate=10).start()
time.sleep(10.0)

# Start FPS counter
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

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

            # Print only once if not already recognized
            if name not in recognized_names:
                print("Recognized:", name)
                recognized_names.add(name)
                
            else:
                break
        else:
            # For unknown faces, check if already seen
            already_seen = False
            break
            for unk in unknown_encodings:
                if face_recognition.compare_faces([unk], encoding, tolerance=0.6)[0]:
                    already_seen = True
                    break

            if not already_seen:
                print("Not found")
                unknown_encodings.append(encoding)

        names.append(name)

    # Draw rectangles and labels
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Facial Recognition", frame)
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
