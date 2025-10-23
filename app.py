import io
import json
from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
from PIL import Image
import os
from fpdf import FPDF
from datetime import datetime
import base64
import tempfile

# --- Inisialisasi Aplikasi Flask & Model YOLO ---
app = Flask(__name__)
model = None
try:
    model = YOLO("best.pt")
    print("Model 'best.pt' berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model: {e}")

# --- Rute Halaman ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tentang")
def about():
    return render_template("about.html")

# --- Rute Prediksi Gambar ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model deteksi tidak dapat dimuat."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada file gambar"}), 400
    file = request.files['image']
    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes))
        results_list = model(pil_image, verbose=False)
        if not results_list:
             parsed_data = []
        else:
            first_result = results_list[0]
            json_data_string = first_result.to_json() 
            parsed_data = json.loads(json_data_string)
        return jsonify(parsed_data), 200
    except Exception as e:
        print(f"Error di /predict: {e}")
        return jsonify({"error": f"Terjadi kesalahan saat memproses gambar: {str(e)}"}), 500

# === RUTE PDF (DIPERBAIKI untuk Chart) ===
@app.route("/download_report", methods=["POST"])
def download_report():
    """Menerima data analisis, gambar deteksi, dan chart, lalu membuat laporan PDF."""
    temp_image_path = None
    temp_chart_path = None # Path untuk file chart sementara
    try:
        data = request.get_json()
        
        grade = data.get("grade", "N/A")
        advice = data.get("advice", "Tidak ada saran.")
        counts = data.get("counts", {})
        total_objects = data.get("total", 0)
        image_data_url = data.get("image_data")
        chart_data_url = data.get("chart_data") # Ambil data chart

        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Laporan Analisis Kualitas Beras", 0, 1, "C")
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "", 10)
        today = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        pdf.cell(0, 10, f"Tanggal Analisis: {today}", 0, 1, "C")
        pdf.ln(10)

        # --- Tambahkan Pie Chart (JIKA ADA) ---
        if chart_data_url:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Distribusi Kualitas", 0, 1, "L")
            
            # Decode gambar chart
            cheader, cencoded = chart_data_url.split(",", 1)
            chart_bytes = base64.b64decode(cencoded)
            
            # Simpan chart ke file temporer
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_c_file:
                temp_c_file.write(chart_bytes)
                temp_chart_path = temp_c_file.name
            
            # Tambahkan chart ke PDF (buat lebih kecil dari gambar utama)
            if temp_chart_path:
                 chart_width = pdf.w / 2 # Setengah lebar halaman
                 pdf.image(temp_chart_path, x=pdf.w / 4, w=chart_width) # Pusatkan chart
            pdf.ln(10)
        # --- Akhir Penambahan Chart ---

        # Ringkasan Hasil (Teks)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Ringkasan Deteksi", 0, 1, "L")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"- Total Objek Terdeteksi: {total_objects}", 0, 1, "L")
        for cls, count in counts.items():
            pdf.cell(0, 8, f"- {cls.replace('-', ' ').title()}: {count} buah", 0, 1, "L")
        pdf.ln(5)

        # Grade Kualitas
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Grade Kualitas", 0, 1, "L")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8, grade, 0, "L")
        pdf.ln(5)

        # Saran
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Saran", 0, 1, "L")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8, advice, 0, "L")
        pdf.ln(10)
        
        # Gambar Hasil Deteksi Utama
        if image_data_url:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Gambar Hasil Deteksi", 0, 1, "L")
            
            header, encoded = image_data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(image_bytes)
                temp_image_path = temp_file.name
            
            if temp_image_path:
                 pdf.image(temp_image_path, x=10, w=pdf.w - 20)

        # Simpan PDF ke buffer
        pdf_buffer = io.BytesIO(pdf.output(dest='S').encode('latin-1'))
        pdf_buffer.seek(0)

        # Kirim file
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name="laporan_deteksi_beras.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        print(f"Error membuat PDF: {e}")
        return jsonify({"error": "Gagal membuat laporan PDF"}), 500
    finally:
        # Hapus KEDUA file temporer
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if temp_chart_path and os.path.exists(temp_chart_path):
            os.remove(temp_chart_path)
# === AKHIR RUTE PDF ===


# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)

