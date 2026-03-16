import cv2, numpy as np, os
from tensorflow.keras.models import load_model

model = load_model("dog_model_finetuned.h5")
classes = sorted(os.listdir("dataset/train"))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    i = np.argmax(pred)

    text = f"{classes[i]} ({round(pred[i]*100,1)}%)"
    cv2.putText(frame, text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Dog Breed Detector", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
