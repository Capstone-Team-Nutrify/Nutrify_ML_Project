import pandas as pd
import numpy as np
import tensorflow as tf

def load_model_and_data():
    model = tf.keras.models.load_model("model/nutrify_multi_model.h5")
    makanan_df = pd.read_csv("data/makanan.csv")

    # Sesuaikan dengan kolom nutrisi aktual
    kolom_nutrisi = [
        "gula", "serat", "protein", "lemak", "karbohidrat", "vitamin_a", "vitamin_c",
        "zat_besi", "kalsium", "natrium", "magnesium", "kolesterol", "kalori", "fosfor",
        "kalium", "zinc", "air", "vitamin_b1", "vitamin_b11", "vitamin_b12", "vitamin_b2",
        "vitamin_b3", "vitamin_b5", "vitamin_b6", "vitamin_d", "vitamin_e", "vitamin_k"
    ]
    
    # Sesuaikan dengan kolom penyakit hasil output model
    penyakit_cols = [
        "Influenza", "Liver", "Diabetes", "Anemia", "Diare", "Batu_Ginjal", "Asma",
        "Asam_Lambung", "Serangan_Jantung", "Asam_Urat", "Radang_Paru_paru", "Jerawat",
        "Hepatitis", "Wasir", "Sinusitis", "Kolesterol", "Usus_Buntu", "Tifus",
        "Osteoporosis", "Malaria", "Alergi_Dingin", "Alergi_Kacang", "Alergi_Seafood",
        "Alergi_Susu", "Alergi_Telur_Ayam", "Alergi_Buah_Beri"
    ]

    return model, makanan_df, kolom_nutrisi, penyakit_cols

def prediksi_status_makanan(nama_makanan, df, fitur_cols, model, label_cols):
    rows = df[df["makanan"].str.lower() == nama_makanan.lower()]
    if rows.empty:
        return {col: None for col in label_cols}  # makanan tidak ditemukan

    baris = rows[fitur_cols].values.astype("float32")
    preds = model.predict(baris)

    hasil = {}
    for i, label in enumerate(label_cols):
        predicted_index = np.argmax(preds[i][0])
        label_map = {0: "Konsumsi Wajar", 1: "Netral", 2: "Waspada"}
        hasil[label] = label_map.get(predicted_index, "Tidak diketahui")
    return hasil
