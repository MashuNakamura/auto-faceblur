#!/usr/bin/env python3
import os
import cv2
import numpy as np
from deepface import DeepFace
import math
import time
from ultralytics import YOLO
import tensorflow as tf

# ============================================================
# SETUP GPU & CUDA PATH
# ============================================================
# Menentukan path lib CUDA untuk XLA compiler TensorFlow.
# Berguna untuk mencegah error libdevice yang sering muncul saat GPU digunakan.
# TF_CPP_MIN_LOG_LEVEL=2 untuk menonaktifkan info log yang tidak perlu dari TensorFlow.
base_dir = os.path.dirname(os.path.abspath(__file__))
cuda_fix_path = os.path.join(base_dir, "cuda_fix")
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_fix_path}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# MENCEGAH TENSORFLOW MEMONOPOLI VRAM
# ============================================================
# Saat DeepFace (TensorFlow) & YOLO (PyTorch) dijalankan bersamaan:
#   - TensorFlow biasanya langsung memblok semua VRAM GPU
#   - set_memory_growth=True agar TF hanya memakai memori yang dibutuhkan, tidak berebut VRAM dengan YOLO
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("INFO: TF Memory Growth Active (Agar tidak berebut dengan YOLO)")
    except RuntimeError as e:
        print(e)

# ============================================================
# CONFIGURATION
# ============================================================
# SKIP_FRAMES : jumlah frame yang dilewati sebelum melakukan face recognition
# THRESHOLD : batas jarak cosine untuk menentukan whitelist face match
# INPUT_RES : resolusi kamera, mempengaruhi akurasi deteksi dan performa
SKIP_FRAMES = 30
THRESHOLD = 0.70
INPUT_RES = (640, 480)

fps = 0
prev_time = time.time()

# ============================================================
# LOAD YOLOv11 MODEL
# ============================================================
# YOLOv11 digunakan untuk deteksi wajah secara cepat.
# model.pt bisa diganti dengan custom model jika sudah dilatih khusus wajah.
# model.to('cuda') agar inferensi YOLO berjalan di GPU untuk performa maksimal.
print("INFO: Loading YOLO...")
model = YOLO("model.pt")
model.to('cuda')
print("INFO: YOLOv11 model loaded on GPU.")

# ============================================================
# LOAD WHITELIST WAJAH
# ============================================================
# Whitelist wajah adalah wajah yang tidak akan diblur.
# DeepFace digunakan untuk ekstraksi embedding wajah dan membandingkan dengan whitelist.
# Semua file dalam folder whitelist akan diproses di awal agar runtime lebih cepat.
print("INFO: Memuat whitelist...")
whitelist_path = "whitelist/"
target_embeddings = []

if not os.path.exists(whitelist_path):
    os.makedirs(whitelist_path)

files = [os.path.join(whitelist_path, f) for f in os.listdir(whitelist_path)
         if os.path.isfile(os.path.join(whitelist_path, f))]

for img_path in files:
    try:
        # Pre-load model DeepFace di awal loop agar tidak lag saat run pertama
        emb = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet512",
            enforce_detection=True
        )
        if emb:
            target_embeddings.append(emb[0]["embedding"])
            print(f"  + Loaded: {os.path.basename(img_path)}")
    except Exception as e:
        print(f"  - Gagal load {img_path}: {e}")

target_embeddings = np.array(target_embeddings)
print(f"INFO: Total {len(target_embeddings)} wajah di whitelist.")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
# get_distance: menghitung jarak cosine antara dua embedding
# blur_face: melakukan Gaussian blur untuk wajah yang tidak whitelisted
# is_close: membandingkan posisi box wajah saat tracking agar tidak perlu rekognisi setiap frame
def get_distance(emb1, emb2):
    a = np.matmul(emb1, emb2)
    b = np.linalg.norm(emb1)
    c = np.linalg.norm(emb2)
    return 1 - (a / (b * c))

def blur_face(face):
    return cv2.GaussianBlur(face, (51, 51), 30)

def is_close(box1, box2, limit=50):
    cx1, cy1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    cx2, cy2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
    dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < limit

# ============================================================
# MAIN LOOP
# ============================================================
# Membaca frame dari webcam, mendeteksi wajah, melakukan face recognition
# dan menerapkan blur pada wajah yang tidak ada di whitelist
# Integrasi YOLO + DeepFace:
#   1. YOLO untuk deteksi bounding box wajah secara real-time
#   2. DeepFace untuk ekstraksi embedding dan pengecekan whitelist
#   3. Tracking dan frame skipping untuk optimasi performa
cap = cv2.VideoCapture(0)
frame_count = 0
tracked_faces = []

print("INFO: Kamera berjalan. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame agar sesuai resolusi input
    sframe = cv2.resize(frame, INPUT_RES)

    # Hitung FPS real-time untuk monitoring performa
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(sframe, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # -------------------------------------------------------
    # DETEKSI WAJAH DENGAN YOLOv11
    # device=0 memastikan GPU digunakan
    # stream=True mengurangi penggunaan memori untuk video loops
    # verbose=False mengurangi log output yang tidak perlu
    # -------------------------------------------------------
    results = model(sframe, stream=True, verbose=False, device=0)

    current_faces_status = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Pastikan koordinat dalam batas frame
            h, w, _ = sframe.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_img = sframe[y1:y2, x1:x2]

            if face_img.size == 0: continue

            is_whitelisted = False
            needs_recognition = (frame_count % SKIP_FRAMES == 0)
            matched_prev_status = None

            # ---------------------------------------------------
            # FRAME SKIPPING & TRACKING
            # Jika wajah sudah terdeteksi di frame sebelumnya,
            # kita tidak perlu rekognisi ulang setiap frame untuk optimasi
            # ---------------------------------------------------
            if not needs_recognition:
                for tf_data in tracked_faces:
                    if is_close([x1, y1, x2, y2], tf_data["box"]):
                        matched_prev_status = tf_data["status"]
                        break
                if matched_prev_status is not None:
                    is_whitelisted = matched_prev_status
                else:
                    needs_recognition = True  # Wajah baru terdeteksi saat skipping

            # ---------------------------------------------------
            # FACE RECOGNITION DENGAN DEEPFACE
            # - Menggunakan embedding untuk membandingkan dengan whitelist
            # - Hanya dilakukan sesuai SKIP_FRAMES agar lebih efisien
            # - Konversi ke RGB untuk kompatibilitas DeepFace
            # ---------------------------------------------------
            if needs_recognition and len(target_embeddings) > 0:
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                try:
                    curr = DeepFace.represent(
                        img_path=face_img_rgb,
                        model_name="Facenet512",
                        enforce_detection=False
                    )
                    if curr:
                        curr_emb = curr[0]["embedding"]
                        # Hitung jarak terdekat ke whitelist
                        dists = [get_distance(t, curr_emb) for t in target_embeddings]
                        min_dist = min(dists)
                        if min_dist <= THRESHOLD:
                            is_whitelisted = True
                except Exception:
                    pass

            # Simpan hasil tracking untuk frame berikutnya
            current_faces_status.append({"box": [x1, y1, x2, y2], "status": is_whitelisted})

            # ---------------------------------------------------
            # VISUALISASI
            # - Blur jika bukan whitelist
            # - Kotak hijau jika whitelist
            # - Label tambahan bisa ditambahkan di sini jika perlu
            # ---------------------------------------------------
            if not is_whitelisted:
                sframe[y1:y2, x1:x2] = blur_face(face_img)
            else:
                cv2.rectangle(sframe, (x1, y1), (x2, y2), (0, 255, 0), 2)

    tracked_faces = current_faces_status
    frame_count += 1

    # Tampilkan frame akhir
    cv2.imshow("YOLOv11 Face Blur", sframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# CLEANUP
# ============================================================
# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()