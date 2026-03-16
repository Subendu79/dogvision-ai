import cv2, numpy as np, os
from tensorflow.keras.models import load_model

model = load_model("dog_model_finetuned.h5")

# Load and prepare image
img = cv2.imread("test.jpg")
if img is None:
    print("Error: test.jpg not found")
    exit()

img = cv2.resize(img, (224,224))

img = img / 255.0
img = np.expand_dims(img, axis=0)

# Load class names
classes = sorted(os.listdir("dataset/train"))

# Predict
pred = model.predict(img)[0]

# Show top-3 predictions
top3 = pred.argsort()[-3:][::-1]

print("\nTop 3 Predictions:")
for i in top3:
    print(classes[i], ":", round(pred[i]*100, 2), "%")
