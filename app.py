from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# DummyModel (opsional jika diperlukan untuk testing)
class DummyModel:
    def predict(self, X):
        return [1 if x[3] > 2 else 0 for x in X]

# Load scaler dan model
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/')
def home():
    # Render file HTML index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data JSON dari permintaan
    data = request.json

    # Konversi data ke DataFrame
    input_data = {
        'Player_Age': [data['player_age']],
        'Player_Weight': [data['player_weight']],
        'Player_Height': [data['player_height']],
        'Previous_Injuries': [data['previous_injuries']],
        'Training_Intensity': [data['training_intensity']],
        'Recovery_Time': [data['recovery_time']]
    }
    input_df = pd.DataFrame(input_data)

    # Normalisasi data
    input_scaled = scaler.transform(input_df)

    # Prediksi menggunakan model
    prediction = model.predict(input_scaled)

    # Tentukan hasil prediksi
    result = "Berisiko Cedera" if prediction[0] == 1 else "Tidak Berisiko Cedera"

    # Kembalikan hasil sebagai JSON
    return jsonify({'prediction': result})

if __name__ == "__main__":
    # Jalankan server Flask
    print("Menjalankan server Flask di http://127.0.0.1:5000")
    app.run(debug=True)
