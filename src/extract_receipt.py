import json
import os
from pathlib import Path
import lmstudio as lms

# 1. Pastikan direktori reports ada
Path("reports").mkdir(parents=True, exist_ok=True)

IMAGE_PATH = Path("data/raw/nota-sample.png")
MODEL_NAME = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-vl-7b-instruct")

# 2. Cek apakah gambar benar-benar ada sebelum lanjut
if not IMAGE_PATH.exists():
    print(f"❌ Error: Gambar tidak ditemukan di {IMAGE_PATH.resolve()}")
    print("💡 Solusi: Pastikan file 'nota-sample.png' sudah ada di folder data/raw/")
    exit(1)

print(f"📸 Memproses gambar: {IMAGE_PATH}")
print(f"🤖 Menggunakan model: {MODEL_NAME}")

try:
    # 3. Siapkan gambar dan model
    image = lms.prepare_image(str(IMAGE_PATH))
    model = lms.llm(MODEL_NAME)
    
    # 4. Buat prompt untuk Vision Model
    chat = lms.Chat()
    chat.add_user_message(
        "Baca nota ini. Ekstrak: merchant, tanggal, item, subtotal, pajak, dan total. "
        "Keluarkan HANYA JSON valid. Jika pajak tidak terlihat, isi 0. Jangan mengarang.",
        images=[image],
    )
    
    # 5. Dapatkan prediksi
    print("⏳ Sedang menganalisis nota (ini mungkin butuh waktu beberapa detik)...")
    prediction = model.respond(chat)
    
    # 6. Bersihkan dan Parse JSON
    content = prediction.content.strip()
    # Hapus markdown block jika model menambahkannya (```json ... ```)
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    result = json.loads(content)
    
    # 7. Simpan dan tampilkan hasil
    output_file = Path("reports/receipt.json")
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print("\n✅ Berhasil! Hasil disimpan di:", output_file.absolute())
    print("-" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-" * 50)
    
except json.JSONDecodeError as e:
    print("❌ Gagal memparse JSON dari output model.")
    print("Output mentah dari model adalah:")
    print(prediction.content)
except Exception as e:
    print(f"❌ Terjadi kesalahan: {e}")
