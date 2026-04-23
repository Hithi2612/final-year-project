import cv2
import os

# Load Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = "harsha"   # Change if needed
sid = "51"

folder = f"dataset/{sid}_{name}"
os.makedirs(folder, exist_ok=True)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Camera not detected")
    exit()

count = 0
print("Capturing face images... Look at camera")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        cv2.imwrite(f"{folder}/{count}.jpg", face_img)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, f"Images: {count}/50", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Dataset Capture", frame)

    if count >= 50:
        break

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()

print("Dataset captured successfully!")
