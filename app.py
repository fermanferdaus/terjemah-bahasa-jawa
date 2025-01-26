import streamlit as st
import cv2
import pytesseract
import numpy as np
import openai
from pdf2image import convert_from_bytes
from gtts import gTTS
import speech_recognition as sr
import os

# Konfigurasi API GPT
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        denoised = cv2.medianBlur(thresh, 3)
        return denoised
    except Exception as e:
        st.error(f"Kesalahan saat preprocessing gambar: {str(e)}")
        return None

# Fungsi untuk membaca dan memproses PDF
def process_pdf(pdf_file):
    try:
        pdf_images = convert_from_bytes(pdf_file.read(), dpi=300)
        all_text = ""
        for idx, page in enumerate(pdf_images):
            st.image(np.array(page), caption=f"Halaman {idx + 1}", use_container_width=True)
            preprocessed_page = preprocess_image(np.array(page))
            if preprocessed_page is not None:
                page_text = pytesseract.image_to_string(preprocessed_page, config='--psm 6', lang="eng+ind")
                all_text += page_text + "\n"
        return all_text
    except Exception as e:
        st.error(f"Kesalahan saat memproses PDF: {str(e)}")
        return ""

# Fungsi untuk menghasilkan TTS
def generate_tts(text, language="id"):
    try:
        tts = gTTS(text, lang=language)
        audio_file = "output.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Kesalahan saat menghasilkan TTS: {str(e)}")
        return None

# Fungsi untuk pengenalan suara
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Silakan berbicara...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="id-ID")
            st.success("Teks yang dikenali: " + text)
            return text
        except sr.UnknownValueError:
            st.error("Tidak dapat mengenali suara.")
        except sr.RequestError as e:
            st.error(f"Kesalahan pada layanan pengenalan suara: {e}")
        return ""

# Judul dan Desain Aplikasi
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
        }
        .sub-title {
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar .css-17eq0hr {
            background-color: #f8f9fa;
        }
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>Aplikasi Penerjemah Bahasa Jawa dan Indonesia dengan OCR</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Dilengkapi dengan Pengenalan Suara</div>", unsafe_allow_html=True)

# Pilihan Input
input_option = st.radio(
    "Pilih metode input:",
    ["Kamera", "Unggah File (Gambar/PDF)", "Pengenalan Suara"],
    horizontal=True,
)

image = None
text = ""

if input_option == "Kamera":
    st.markdown("### Ambil Gambar dengan Kamera")
    captured_image = st.camera_input("Klik untuk membuka kamera")
    if captured_image is not None:
        file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif input_option == "Unggah File (Gambar/PDF)":
    st.markdown("### Unggah File (Gambar/PDF)")
    uploaded_file = st.file_uploader("Pilih file untuk diunggah", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = process_pdf(uploaded_file)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif input_option == "Pengenalan Suara":
    # Tombol untuk Mulai Rekam dan Reset
    st.markdown("### Rekam Suara")
    if "recognized_text" not in st.session_state:
        st.session_state["recognized_text"] = ""

    if st.button("Mulai Rekam"):
        st.session_state["recognized_text"] = recognize_speech()

    if st.button("Reset"):
        st.session_state["recognized_text"] = ""

    text = st.session_state["recognized_text"]

if image is not None:
    st.markdown("### Gambar yang Diproses")
    st.image(image, caption="Gambar Asli", use_container_width=True)
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is not None:
        text = pytesseract.image_to_string(preprocessed_image, config='--psm 6', lang="eng+ind")

if text.strip():
    st.markdown("### Teks yang Dikenali")
    st.text_area("Hasil:", text, height=200)

    if st.button("Terjemahkan"):
        try:
            prompt = f"""
                Kamu adalah penerjemah profesional yang sangat ahli dalam Bahasa Jawa dan Bahasa Indonesia. 
                Berikut adalah teks yang perlu diterjemahkan:
                
                {text}
                
                Terjemahkan teks ini ke bahasa yang paling sesuai dengan mempertahankan arti, konteks, dan nuansa budaya.
                """
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
            )
            translation = response['choices'][0]['message']['content'].strip()
            st.success("Hasil Terjemahan:")
            st.markdown(f"### {translation}")

            # TTS untuk hasil terjemahan
            tts_file = generate_tts(translation, language="id")
            if tts_file:
                audio_bytes = open(tts_file, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
else:
    st.info("Unggah file, ambil gambar, atau gunakan pengenalan suara untuk memulai.")
