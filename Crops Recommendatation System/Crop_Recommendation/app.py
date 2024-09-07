from flask import Flask, request, render_template
import numpy as np
import pickle

# Correct file paths to the pickle files
model_path = r'Crop_Recommendation\model.pkl'
standscaler_path = r'Crop_Recommendation\standscaler.pkl'
minmaxscaler_path = r'Crop_Recommendation\minmaxscaler.pkl'

# Load the model and scalers
model = pickle.load(open(model_path, 'rb'))  # The machine learning model
sc = pickle.load(open(standscaler_path, 'rb'))  # Standard scaler
mx = pickle.load(open(minmaxscaler_path, 'rb'))  # MinMax scaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Create feature array
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply the MinMaxScaler and StandardScaler
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)

        # Predict the crop
        prediction = model.predict(sc_mx_features)

        # Dictionary to map predicted values to crop names
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        # Get the crop name from the prediction
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    except Exception as e:
        result = f"An error occurred: {e}"

    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
