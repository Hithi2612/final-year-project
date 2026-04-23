import cv2, os, numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, ids = [], []
test_image_paths = []

for folder in os.listdir("dataset"):
    try:
        sid = int(folder.split("_")[0])
    except:
        continue
        
    for img in os.listdir(f"dataset/{folder}"):
        path = f"dataset/{folder}/{img}"
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (200,200))
        gray = cv2.equalizeHist(gray)

        faces.append(gray)
        ids.append(sid)

        test_image_paths.append(path)

recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")

correct = 0
total = 0

# FIXED FUNCTION
def get_actual_label(path):
    folder = os.path.basename(os.path.dirname(path))
    return int(folder.split("_")[0])

for imagePath in test_image_paths:
    gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (200,200))
    
    label, confidence = recognizer.predict(gray)
    actual_label = get_actual_label(imagePath)

    total += 1
    if label == actual_label:
        correct += 1

accuracy = (correct / total) * 100
print("Test Accuracy:", accuracy)

print("Model trained successfully")
