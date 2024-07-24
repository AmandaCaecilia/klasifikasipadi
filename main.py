import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


# Fungsi untuk memproses gambar
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img = img.convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Klasifikasi Penyakit Daun padi", page_icon=":leaves:", layout="centered")

# Tema warna hijau
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e8f5e9;
    }
    header, .sidebar .sidebar-content {
        background: linear-gradient(90deg, #43a047, #66bb6a);
    }
    .sidebar .sidebar-content {
        background: #43a047;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul aplikasi
st.title("Klasifikasi Penyakit Daun padi")
st.markdown("Unggah gambar daun padi untuk mendeteksi penyakitnya.")

# Memuat model
model_path = 'fold_4_model_D-2_epoch_50.h5'  # Ganti dengan path model Anda
model = load_model(model_path)

# Mengunggah gambar
uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Gambar yang diunggah', use_column_width=True)
    processed_image = preprocess_image(uploaded_file, target_size=(224, 224))
    st.write("")
    st.write("Mendeteksi...")
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Menampilkan hasil prediksi
    class_names = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]  # Ganti dengan nama kelas yang sesuai
    st.write(f"Hasil Prediksi: *{class_names[predicted_class]}*")

st.markdown(
    """
    <style>
    .css-1cpxqw2, .st-c6, .st-ci {
        color: #388e3c;
    }
    .css-10trblm {
        background-color: #a5d6a7;
    }
    </style>
    """,
    unsafe_allow_html=True
)
