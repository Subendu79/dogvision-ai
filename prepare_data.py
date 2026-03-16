import os, shutil

IMG_DIR = "images/Images"
ANN_DIR = "annotations/Annotation"
OUT_DIR = "dataset/train"

os.makedirs(OUT_DIR, exist_ok=True)

for breed in os.listdir(IMG_DIR):
    src_breed = os.path.join(IMG_DIR, breed)
    if not os.path.isdir(src_breed):
        continue

    dst_breed = os.path.join(OUT_DIR, breed)
    os.makedirs(dst_breed, exist_ok=True)

    for img in os.listdir(src_breed):
        src = os.path.join(src_breed, img)
        dst = os.path.join(dst_breed, img)
        shutil.copy(src, dst)

print("DONE")
