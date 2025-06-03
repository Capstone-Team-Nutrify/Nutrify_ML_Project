from flask import Flask, request, jsonify
from model_utils import load_model_and_data, prediksi_status_makanan

# Init Flask
app = Flask(__name__)

# Load model & data only once
model, makanan_df, kolom_nutrisi, penyakit_cols = load_model_and_data()

@app.route("/predict_makanan", methods=["POST"])
def predict_makanan():
    req = request.json
    nama_makanan = req.get("makanan")
    if not nama_makanan:
        return jsonify({"error": "Parameter 'makanan' wajib diisi"}), 400

    hasil = prediksi_status_makanan(nama_makanan, makanan_df, kolom_nutrisi, model, penyakit_cols)
    return jsonify({"makanan": nama_makanan, "status_penyakit": hasil})

if __name__ == "__main__":
    app.run(debug=True)
