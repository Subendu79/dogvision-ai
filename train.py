
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

print("Training script started")

img_size = 224
batch = 32

gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.25,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    validation_split=0.2
)


train = gen.flow_from_directory(
    "dataset/train",
    target_size=(img_size,img_size),
    batch_size=batch,
    class_mode="categorical",
    subset="training"
)

val = gen.flow_from_directory(
    "dataset/train",
    target_size=(img_size,img_size),
    batch_size=batch,
    class_mode="categorical",
    subset="validation"
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

# Freeze most of the model
for layer in base.layers[:-60]:
    layer.trainable = False
for layer in base.layers[-60:]:
    layer.trainable = True



x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
output = Dense(train.num_classes, activation="softmax")(x)

model = Model(base.input, output)

from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


model.fit(train, validation_data=val, epochs=8)


model.save("dog_model_finetuned.h5")


