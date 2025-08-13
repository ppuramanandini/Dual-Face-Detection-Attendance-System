from collections import defaultdict
import time
...
seen_once = set()                 # print only once per name
last_seen = defaultdict(float)    # throttle prints by cooldown seconds
COOLDOWN = 5.0                    # seconds

...

while True:
    vs = VideoStream(src=VIDEO_SRC, framerate=10).start()
    frame = imutils.resize(frame, width=500)

    # face_recognition expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "person not found"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

            # Print only once per person (or once per COOLDOWN)
            now = time.time()
            if (name not in seen_once) or (now - last_seen[name] > COOLDOWN):
                print("Recognized:", name)
                seen_once.add(name)
                last_seen[name] = now

        names.append(name)

    # Draw boxes
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    fps.update()
