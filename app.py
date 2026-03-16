from flask import Flask, render_template, request
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("dog_model_finetuned.h5")

classes = sorted(os.listdir("dataset/train"))

breed_info = {
"Siberian_husky":{
"img":"https://upload.wikimedia.org/wikipedia/commons/5/54/Siberian_Husky.jpg",
"desc":"A strong and energetic working dog known for endurance and friendly nature."
},

"German_shepherd":{
"img":"https://upload.wikimedia.org/wikipedia/commons/3/3d/German_Shepherd.jpg",
"desc":"Highly intelligent and loyal breed often used in police and military work."
},

"golden_retriever":{
"img":"https://upload.wikimedia.org/wikipedia/commons/8/8b/Golden_Retriever.jpg",
"desc":"Friendly, intelligent and devoted family dog."
}
}

def prepare_image(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET","POST"])
def index():

    results = None

    if request.method == "POST":

        file = request.files["image"]
        img = Image.open(file.stream)

        img = prepare_image(img)

        pred = model.predict(img)[0]

        top3 = pred.argsort()[-3:][::-1]

        results = []

        for i in top3:

            breed = classes[i]
            confidence = round(pred[i]*100,2)

            results.append({
                "breed":breed,
                "confidence":confidence,
                "info":breed_info.get(breed)
            })

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)