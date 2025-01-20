# app.py
import streamlit as st
import pandas as pd
import joblib
import time
import json
from preprocessing import *
from train_model import train_model as tm
import plotly.express as px


# Streamlit App
st.title("Prediksi Kenaikan Jabatan Karyawan")
st.write("Aplikasi ini memprediksi apakah seorang karyawan akan dipromosikan berdasarkan data yang diunggah.")

def predict():
    # Halaman Prediksi
    st.header("Prediksi Kenaikan Jabatan")

    
    # Load the trained model and preprocessing objects
    try:
        model = joblib.load('random_forest_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        st.error("Model belum dilatih. Silakan buka halaman 'Train Model' untuk melatih model terlebih dahulu.")
        st.stop()
    
    # Upload file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if st.button("Predict Dataset"):

        if uploaded_file is None:
            st.warning("Silakan unggah file CSV terlebih dahulu.")
            return

        with st.status("Proses Predict dataset...", expanded=True) as status:
            st.write("Load dataset...")
            # Load the uploaded data
            new_data = pd.read_csv(uploaded_file)
            time.sleep(2)

            st.write("Preprocessing data...")
            # Preprocess the new data
            new_data = handle_missing_values(new_data)
            new_data = casefolding(new_data)
            for column in label_encoders.keys():
                new_data[column] = label_encoders[column].transform(new_data[column])
            numerical_cols = ['age', 'length_of_service', 'avg_training_score', 'previous_year_rating']
            new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])
            new_data, _ = feature_engineering(new_data, new_data)
            new_data, _ = drop_unnecessary_columns(new_data, new_data)
            time.sleep(2)

            st.write("Predict data...")
            try:
                # Make predictions
                predictions = model.predict(new_data)
                new_data['is_promoted'] = predictions
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi model: {e}")
            time.sleep(1)
            st.write("Done!")
        
        new_data = pd.DataFrame(new_data)

        # Hitung jumlah prediksi promosi vs tidak promosi
        promoted_count = new_data['is_promoted'].value_counts().get(1, 0)  # Jumlah dipromosikan
        not_promoted_count = new_data['is_promoted'].value_counts().get(0, 0)  # Jumlah tidak dipromosikan

        # Data untuk pie chart
        pie_data = {
            'Kategori': ['Dipromosikan', 'Tidak Dipromosikan'],
            'Jumlah': [promoted_count, not_promoted_count]
        }
        pie_df = pd.DataFrame(pie_data)

        # Buat pie chart
        fig = px.pie(pie_df, values='Jumlah', names='Kategori', 
                    title='Distribusi Prediksi Promosi Karyawan',
                    color_discrete_sequence=px.colors.qualitative.Set3)

        # Tampilkan pie chart di Streamlit
        st.plotly_chart(fig)
        
        # Display the predictions
        st.write("Hasil Prediksi:")
        st.write(new_data)
        
        # Download the predictions as a CSV file
        st.download_button(
            label="Download Hasil Prediksi",
            data=new_data.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv',
        )

def train_model():

    def load_results_from_json(filename="results.json"):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    results = load_results_from_json()
    if results:
        st.success("Model sudah dilatih sebelumnya. Berikut adalah hasilnya:")
        
        # Tampilkan hasil training dari file JSON
        st.metric(label="Training Accuracy", value=f"{results['training_accuracy']:.2%}")
        st.metric(label="Testing Accuracy", value=f"{results['testing_accuracy']:.2%}")
        st.subheader("Classification Report")
        st.code(results['classification_report'], language='text')
    
    # Halaman Train Model
    st.header("Train Model")
    
    # Informasi dataset
    st.write("Dataset yang digunakan untuk training:")
    st.write("- Training data: `/dataset/train.csv`")
    st.write("- Testing data: `/dataset/test.csv`")

    def save_results_to_json(results, filename="results.json"):
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)

    # Tombol untuk memulai training
    if st.button(results and "Update Model" or "Train Model"):
        
        with st.status("Proses Training model...", expanded=True) as status:
            st.write("Load dataset...")
            time.sleep(2)
            st.write("Preprocessing data...")
            time.sleep(2)
            st.write("Training model...")
            try:
                results = tm('train.csv', 'test.csv')
                with open("results.json", "w") as f:
                    json.dump(results, f, indent=4)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model: {e}")
            time.sleep(1)
            st.write("Done!")

        
        st.success("Model berhasil dilatih dan disimpan!")

        st.metric(label="Training Accuracy", value=f"{results['training_accuracy']:.2%}")
        st.metric(label="Testing Accuracy", value=f"{results['testing_accuracy']:.2%}")
            
        st.subheader("Classification Report")
        st.code(results['classification_report'], language='text')
        
    

pages = {
    "Main Menu": [
        st.Page(predict, title="Predict Dataset"),
        st.Page(train_model, title="Train Model"),
    ]
}

pg = st.navigation(pages)
pg.run()

