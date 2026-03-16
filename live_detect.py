from flask import Flask, request, render_template_string
import cv2, numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

model = load_model("dog_model_finetuned.h5")
classes = sorted(__import__("os").listdir("dataset/train"))

app = Flask(__name__)

HTML = """
<html>
<head><title>Dog Breed Detector</title></head>
<body style="text-align:center;">
<h2>Dog Breed Detector 🐶</h2>
<form method="post" enctype="multipart/form-data">
<input type="file" name="photo" accept="image/*" capture="camera">
<br><br>
<button type="submit">Detect Breed</button>
</form>
{% if results %}
<h3>Top Predictions:</h3>
<ul style="list-style:none;">
{% for r in results %}
<li>{{ r }}</li>
{% endfor %}
</ul>
{% endif %}


</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    results = None

    if request.method == "POST":
        file = request.files["photo"]
        img = Image.open(file.stream).resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]
        top3 = pred.argsort()[-3:][::-1]

        results = []
        for i in top3:
            results.append(f"{classes[i]} : {round(pred[i]*100, 2)} %")

    return render_template_string(HTML, results=results)


app.run(host="0.0.0.0", port=5000)
