from flask import Flask, request, render_template, redirect, Markup
import requests
import io
import os
import joblib
import torch
from torchvision import transforms
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
from PIL import Image
from utils.disease_model import ResNet9
from utils.disease import disease_dic


# loading pipeline files through joblib
model_yield = joblib.load(open("model_files/crop_yield_prediction.joblib", "rb"))
model_cost = joblib.load(open("model_files/price_prediction_icrisat.joblib", "rb"))
model_soil = joblib.load(open("model_files/soil_health.joblib","rb"))

# loading dl model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'model_files/plant-disease-model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open((img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

app = Flask(__name__)


# =================Landing Pages========================
@app.route("/")
def home():
    title='I-Farm'
    return render_template("index.html", title=title)


@app.route("/production-prediction", methods=["POST", "GET"])
def prod_predict():
    title='Yield Prediction'
    return render_template("yield_predict.html", title=title)


@app.route("/price-prediction")
def price_predict():
    title='Price Prediction'
    return render_template("price_predict.html",title=title)


@app.route("/soil-prediction", methods=["POST", "GET"])
def soil_predict():
    title='Soil Health'
    return render_template("soil_predict.html",title=title)


@app.route("/disease-prediction", methods=["POST", "GET"])
def disease_predict():
    title='Disease Prediction'
    return render_template("disease_predict.html", title=title)

@app.route("/About-us", methods = ['GET'])
def about_us():
    title='About Us'

    return render_template("AboutUs.html", title=title)

# ========= Result pages =======
@app.route("/production-result", methods=["POST", "GET"])
def prod_result():

    title='Yield Result'
    
    if request.method == "POST":
        state = request.form["State"]
        crop = request.form["Crop"]
        season = request.form["Season"]
        year = int(request.form["Year"])
        area = float(request.form["Area"])
        rainfall = float(request.form["Rainfall"])

    production = model_yield.predict([[state, year, season, crop, area, rainfall]])[0]
    yield_on = production / area

    production = round(production, 2)
    yield_on = round(yield_on, 2)

    return render_template(
        "yield_result.html", production=production, yield_on=yield_on,title=title
    )


@app.route("/price-result", methods=["POST", "GET"])
def price_result():

    title='Price Result'

    if request.method == "POST":
        state = request.form["State"]
        crop = request.form["Crop"]
        year = int(request.form["Year"])

    
    cost = model_cost.predict([[state, crop, year]])[0]
    cost = round(cost, 2)

    return render_template(
        "price_result.html",
        cost=cost, title=title
    )

@app.route("/soil-result",  methods=["POST", "GET"])
def soil_result():
    
    title='Soil Result'

    if request.method == "POST":
        crop = request.form.get("Crop")
        temperature = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"]) 
        rainfall = float(request.form["Rainfall"])
        ph = float(request.form["Ph"])

    result = model_soil.predict([[crop,temperature,humidity,rainfall,ph]])
    
    N = int(result[0][0])
    P = int(result[0][1])
    K = int(result[0][2])

    return render_template("soil_result.html", N=N, P=P, K=K,title=title)


@app.route("/disease-result", methods=['POST','GET'])
def disease_result():
    title='Disease Detection'
    
    base_path = os.path.dirname("__file__")
    target = os.path.join(base_path, 'static/uploads/')
    
    if not os.path.isdir(target):
        os.mkdir(target)
    
    for file in request.files.getlist('file'):

        file_name = secure_filename(file.filename)
        destination = "/".join([target, file_name])
        file.save(destination)
    
    img = open(destination,"rb")
    img.read()

    prediction = predict_image(img)
    prediction = Markup(str(disease_dic[prediction]))

    return render_template('disease_result.html', prediction=prediction, title=title)


if __name__ == "__main__":
    app.run(debug=False)
