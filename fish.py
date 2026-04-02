import streamlit as st
from PIL import Image
import os
import gdown
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Ikan",
    page_icon="🐟",
    layout="wide"
)

# ==============================================================================
# 2. KONSTANTA GLOBAL
# (Saran #3, #7: Pindahkan magic number & class_names ke level modul)
# ==============================================================================
IMG_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 5
MIN_MODEL_SIZE_BYTES = 1_000_000  # 1 MB

CLASS_NAMES = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel',
    'Red Mullet', 'Red Sea Bream', 'Sea Bass',
    'Shrimp', 'Striped Red Mullet', 'Trout'
]

LAYER_CANDIDATES = [
    "conv5_block3_out",
    "conv5_block3_3_conv",
    "top_conv",
    "block5_conv3",
    "activation_49"
]

# (Saran keamanan: Gunakan st.secrets untuk GDRIVE_FILE_ID di production)
# GDRIVE_FILE_ID = st.secrets["GDRIVE_FILE_ID"]
GDRIVE_FILE_ID = '1vAijkvT2TrxjSS-GvpNfZLy3he4aGGv9'

# ==============================================================================
# 3. MODEL LOADING
# (Saran #2: Tambahkan validasi integritas file model)
# ==============================================================================
@st.cache_resource
def load_model_from_gdrive():
    """
    Downloads and loads a Keras model in .h5 format from Google Drive.
    Validates file integrity before loading.
    Caches the model so it doesn't reload on every interaction.
    """
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    output = 'model.h5'

    # Cek apakah file ada DAN ukurannya valid (bukan file corrupt/kosong)
    file_is_valid = (
        os.path.exists(output) and
        os.path.getsize(output) >= MIN_MODEL_SIZE_BYTES
    )

    if not file_is_valid:
        if os.path.exists(output):
            os.remove(output)  # Hapus file corrupt sebelum download ulang
        with st.spinner("Mengunduh model dari Google Drive... (mungkin memerlukan beberapa saat)"):
            gdown.download(url, output, quiet=False)

    model = tf.keras.models.load_model(output, compile=False)
    return model

# ==============================================================================
# 4. IMAGE PROCESSING AND PREDICTION
# (Saran #3: class_names sudah dipindah ke konstanta global CLASS_NAMES)
# ==============================================================================
def transform_image_for_prediction(pil_img):
    """
    Preprocesses a PIL image for prediction with the ResNet50 model.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    pil_img = pil_img.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.resnet50.preprocess_input(img_array)
    return processed_img

def predict(pil_img, model):
    """
    Makes a prediction and returns the top 3 results with their confidence scores.
    """
    processed_img = transform_image_for_prediction(pil_img)
    predictions = model.predict(processed_img)[0]

    top_3_indices = np.argsort(predictions)[-3:][::-1]
    top_3_results = [(CLASS_NAMES[i], float(predictions[i])) for i in top_3_indices]
    return top_3_results

# ==============================================================================
# 5. GRAD-CAM VISUALIZATION
# (Saran #4: Ganti cm.get_cmap yang deprecated)
# (Saran #5: Error handling lebih informatif)
# ==============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given image and model layer.
    """
    grad_model = tf.keras.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        if isinstance(preds, list):
            preds = preds[0]

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)

    return heatmap.numpy()

def display_gradcam(pil_img, heatmap, alpha=0.5):
    """
    Superimposes the Grad-CAM heatmap on the original image.
    """
    img = tf.keras.utils.img_to_array(pil_img)

    heatmap_uint8 = np.uint8(255 * heatmap)

    # (Saran #4: Ganti cm.get_cmap yang deprecated dengan plt.colormaps)
    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap_img = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap_img = jet_heatmap_img.resize((img.shape[1], img.shape[0]))
    jet_heatmap_arr = tf.keras.utils.img_to_array(jet_heatmap_img)

    superimposed_img = jet_heatmap_arr * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    return superimposed_img

# ==============================================================================
# 6. SALIENCY MAP (FALLBACK)
# (Saran #6: Ganti np.kron dengan resize yang proper)
# ==============================================================================
def generate_saliency_map(model, img_array):
    """
    Creates a saliency map as a fallback visualization.
    """
    img_array = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_array)
        predictions = model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[0][top_pred_index]

    gradients = tape.gradient(top_class_channel, img_array)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]
    saliency = (saliency - tf.reduce_min(saliency))
    saliency /= (tf.reduce_max(saliency) + 1e-9)

    return saliency.numpy()

def display_saliency_map(saliency, original_img):
    """
    Displays the saliency map overlaid on the original image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(original_img.resize(IMG_SIZE))

    # (Saran #6: Ganti np.kron dengan resize PIL yang akurat)
    saliency_pil = Image.fromarray((saliency * 255).astype(np.uint8)).resize(IMG_SIZE)
    saliency_resized = np.array(saliency_pil) / 255.0

    ax.imshow(saliency_resized, cmap='jet', alpha=0.5)
    ax.axis('off')

    return fig

# ==============================================================================
# 7. VALIDASI FILE UPLOAD
# (Saran #8: Validasi ukuran dan tipe gambar)
# ==============================================================================
def validate_uploaded_file(uploaded_file):
    """
    Validates file size. Returns (is_valid, error_message).
    """
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"Ukuran file terlalu besar ({file_size_mb:.1f} MB). Maksimal {MAX_FILE_SIZE_MB} MB."
    return True, None

# ==============================================================================
# 8. BANNER
# ==============================================================================
def display_banner():
    BANNER_URL = "banner.png"
    if os.path.exists(BANNER_URL):
        st.image(BANNER_URL, use_container_width=True)

# ==============================================================================
# 9. HOMEPAGE
# ==============================================================================
def homepage():
    display_banner()
    st.markdown("## Selamat Datang di Aplikasi Klasifikasi Ikan")
    st.markdown("Aplikasi ini menggunakan model *Deep Learning* untuk mengklasifikasikan 9 jenis ikan dan memberikan visualisasi menggunakan Grad-CAM untuk interpretasi.")

    with st.expander("📚 **Tentang Aplikasi Ini**"):
        st.markdown("""
        **Fitur Utama:**
        - Klasifikasi 9 jenis ikan laut
        - Visualisasi daerah penting dengan Grad-CAM
        - Tampilkan top 3 prediksi
        - Riwayat prediksi
        - Antarmuka pengguna yang mudah digunakan

        **Cara Penggunaan:**
        1. Pindah ke halaman **Prediction** di sidebar
        2. Unggah gambar ikan (maks. 5 MB)
        3. Lihat hasil klasifikasi dan visualisasi
        4. Tinjau riwayat di halaman **History**
        """)

    st.markdown("### Contoh Jenis Ikan yang Dikenali")
    st.write("Berikut adalah 9 jenis ikan yang dapat dikenali oleh sistem:")

    image_files = {
        "Black Sea Sprat": "blackseasprat.png",
        "Gilt-Head Bream": "giltheadbream.jpeg",
        "Horse Mackerel": "horsemackerel.png",
        "Red Mullet": "redmullet.png",
        "Red Sea Bream": "redseabream.jpeg",
        "Sea Bass": "seabass.jpeg",
        "Shrimp": "shrimp.png",
        "Striped Red Mullet": "stripedredmullet.png",
        "Trout": "trout.png"
    }

    cols = st.columns(3)
    for i, (caption, filename) in enumerate(image_files.items()):
        if os.path.exists(filename):
            with cols[i % 3]:
                st.image(filename, caption=caption)

    st.markdown("---")
    st.info("Klik **Prediction** di sidebar untuk mengunggah gambar dan melihat hasil prediksi.")

# ==============================================================================
# 10. HISTORY PAGE
# ==============================================================================
def history_page():
    display_banner()
    st.markdown("## 📜 Prediction History")
    st.write("Tinjau hasil klasifikasi sebelumnya")

    if 'history' not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        st.info("Belum ada riwayat prediksi. Lakukan prediksi di halaman Prediction.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Riwayat Prediksi")
    with col2:
        if st.button("🧹 Hapus History", use_container_width=True, type="primary"):
            st.session_state.history = []
            st.rerun()

    for i, record in enumerate(reversed(st.session_state.history)):
        with st.expander(f"**{record['timestamp']}** - {record['image_name']}", expanded=i == 0):
            col_img, col_data = st.columns([0.3, 0.7])

            with col_img:
                st.image(record['thumbnail'], caption="Gambar yang Diunggah", use_container_width=True)

            with col_data:
                confidence = float(record['confidence'])
                st.subheader(f"Prediksi: **{record['prediction']}**")
                st.progress(confidence)
                st.caption(f"Confidence: {confidence * 100:.2f}%")

                st.markdown("**Top 3 Prediksi:**")
                df = pd.DataFrame(record['top_predictions'], columns=['Class', 'Confidence'])
                df['Confidence_val'] = df['Confidence']

                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Class": st.column_config.TextColumn("Class", width="medium"),
                        "Confidence": st.column_config.TextColumn("Confidence"),
                        "Confidence_val": st.column_config.ProgressColumn(
                            "Confidence Bar",
                            format="%.2f%%",
                            min_value=0,
                            max_value=1,
                        )
                    }
                )

                if 'visualization' in record and record['visualization']:
                    st.markdown("**Visualisasi Model:**")
                    st.image(record['visualization'], use_container_width=True)

# ==============================================================================
# 11. PREDICTION PAGE
# (Saran #1: Perbaikan bug duplikat history dengan image hash)
# (Saran #5: Error handling lebih informatif)
# (Saran #8: Validasi file upload)
# ==============================================================================
def prediction_page():
    display_banner()
    st.markdown("### Unggah Gambar Ikan untuk Klasifikasi")
    st.write("Unggah gambar ikan (JPG/PNG, maks. 5 MB) untuk diklasifikasikan oleh model.")

    model = load_model_from_gdrive()

    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    visualization_created = False
    superimposed_image = None

    if uploaded_file is not None:
        # (Saran #8: Validasi ukuran file)
        is_valid, error_msg = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(error_msg)
            st.stop()

        pil_image = Image.open(uploaded_file).convert("RGB")

        # (Saran #1: Hitung hash gambar untuk mencegah duplikat history)
        image_hash = hash(uploaded_file.getvalue())

        col1, col2 = st.columns([0.9, 1.1])

        with col1:
            st.image(pil_image, caption="Gambar yang Diunggah", use_container_width=True)

        with col2:
            with st.spinner('Mengklasifikasi...'):
                top_3_results = predict(pil_image, model)

            predicted_class, confidence = top_3_results[0]

            st.write(f"### Hasil Prediksi: **{predicted_class}**")
            st.write(f"Tingkat Confidence: **{confidence * 100:.2f}%**")

            st.write("---")
            st.write("#### Top 3 Prediksi")
            df_results = pd.DataFrame(top_3_results, columns=['Kelas Ikan', 'Confidence'])
            st.bar_chart(df_results.set_index('Kelas Ikan'))

        # --- Visualization Section ---
        st.write("---")
        st.markdown("### Visualisasi Model")
        st.write("Heatmap menunjukkan area pada gambar yang paling memengaruhi keputusan model.")

        img_array_for_vis = np.expand_dims(
            tf.keras.utils.img_to_array(pil_image.resize(IMG_SIZE)),
            axis=0
        )

        with st.spinner("Membuat visualisasi..."):
            # Coba Grad-CAM dengan berbagai layer
            for layer_name in LAYER_CANDIDATES:
                try:
                    model.get_layer(layer_name)
                    heatmap = make_gradcam_heatmap(img_array_for_vis, model, layer_name)
                    superimposed_image = display_gradcam(pil_image, heatmap)
                    st.image(
                        superimposed_image,
                        caption=f'Grad-CAM Heatmap (Layer: {layer_name})',
                        use_container_width=True
                    )
                    st.success(f"Berhasil membuat Grad-CAM dengan layer: {layer_name}")
                    visualization_created = True
                    break
                except ValueError:
                    # Layer tidak ditemukan, coba layer berikutnya
                    continue
                except Exception as e:
                    # (Saran #5: Tampilkan info error yang lebih informatif)
                    st.warning(f"Layer `{layer_name}` gagal: {type(e).__name__}: {e}")
                    continue

            # Fallback ke saliency map
            if not visualization_created:
                try:
                    st.warning("Grad-CAM tidak berhasil. Mencoba saliency map sebagai alternatif...")
                    saliency = generate_saliency_map(model, img_array_for_vis)
                    fig = display_saliency_map(saliency, pil_image)
                    st.pyplot(fig)
                    st.info("Saliency Map: Area berwarna menunjukkan pengaruh terbesar pada keputusan model.")
                    visualization_created = True
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    superimposed_image = Image.open(buf)
                except Exception as e:
                    st.error(f"Gagal membuat visualisasi: {type(e).__name__}: {e}")

            if visualization_created:
                st.info("""
                **Interpretasi:**
                - Area **merah/panas**: Sangat memengaruhi prediksi
                - Area **biru/dingin**: Kurang memengaruhi prediksi
                """)

        # --- Simpan ke History (hanya jika belum ada) ---
        if 'history' not in st.session_state:
            st.session_state.history = []

        # (Saran #1: Cek duplikat berdasarkan hash gambar sebelum menyimpan)
        already_saved = any(
            r.get('image_hash') == image_hash
            for r in st.session_state.history
        )

        if not already_saved:
            thumbnail = pil_image.copy()
            thumbnail.thumbnail((200, 200))

            vis_bytes = None
            if visualization_created and superimposed_image:
                buf = BytesIO()
                if isinstance(superimposed_image, Image.Image):
                    superimposed_image.save(buf, format='PNG')
                else:
                    plt.savefig(buf, format='png')
                vis_bytes = buf.getvalue()

            history_record = {
                'image_hash': image_hash,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_name': uploaded_file.name,
                'thumbnail': thumbnail,
                'prediction': top_3_results[0][0],
                'confidence': float(top_3_results[0][1]),
                'top_predictions': top_3_results,
                'visualization': vis_bytes
            }

            st.session_state.history.append(history_record)

# ==============================================================================
# 12. MAIN APP LOGIC
# ==============================================================================
def main():
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_container_width=True)

    st.sidebar.title("Navigasi")

    pages = {
        "Home": homepage,
        "Prediction": prediction_page,
        "History": history_page,
    }

    selection = st.sidebar.radio("Pindah ke Halaman", list(pages.keys()))
    pages[selection]()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tentang Aplikasi")
    st.sidebar.info("""
    Aplikasi Klasifikasi Ikan
    - Dibuat dengan Streamlit dan TensorFlow
    - Model ResNet50 yang telah dilatih ulang
    - [GitHub Repository]()
    - [Google Colaboratory]()
    """)

if __name__ == "__main__":
    main()