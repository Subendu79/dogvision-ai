import cv2

index = 0
found = []

while True:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        break
    found.append(index)
    cap.release()
    index += 1

print("Available camera indexes:", found)
