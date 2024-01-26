# Import library dan module yang diperlukan
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set level log TensorFlow untuk hanya menampilkan pesan yang penting
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

# Memuat model prediksi huruf dari file 'smnist.h5'
model = load_model('smnist.h5')

# Menginisialisasi modul deteksi tangan dari Mediapipe
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Membuka kamera (camera index 0)
cap = cv2.VideoCapture(0)

# Membaca frame pertama dari kamera
_, frame = cap.read()

# Mendapatkan dimensi tinggi, lebar, dan jumlah saluran warna dari frame
h, w, c = frame.shape

# Variabel untuk menghitung jumlah gambar yang diambil
img_counter = 0

# List karakter huruf yang akan diprediksi
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Loop utama untuk memproses setiap frame secara terus-menerus
while True:
    # Membaca frame dari kamera
    _, frame = cap.read()

    # Memeriksa apakah frame kosong
    if frame is None:
        print("Error: Empty frame")
        break

    # Menangkap input keyboard (ESC untuk keluar)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    
    # Mengubah format warna frame ke RGB
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Memproses frame menggunakan modul deteksi tangan dari Mediapipe
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    
    # Jika terdeteksi tangan pada frame
    if hand_landmarks:
        for handLMs in hand_landmarks:
            # Inisialisasi variabel batas koordinat tangan
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            
            # Mendapatkan koordinat tangan (landmarks)
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                
                # Memperbarui batas koordinat tangan
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            
            # Menyesuaikan batas untuk memastikan berada dalam rentang yang valid
            y_min = max(0, y_min - 20)
            y_max = min(h, y_max + 20)
            x_min = max(0, x_min - 20)
            x_max = min(w, x_max + 20)

            # Memeriksa validitas region of interest
            if x_min < x_max and y_min < y_max:
                # Menggambar kotak di sekitar region of interest
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Segmentasi area tangan
                analysisframe = frame[y_min:y_max, x_min:x_max]
                analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
                analysisframe = cv2.resize(analysisframe, (28, 28))

                # Mendapatkan data piksel dan mengonversi ke format yang sesuai dengan model
                nlist = []
                rows, cols = analysisframe.shape
                for i in range(rows):
                    for j in range(cols):
                        k = analysisframe[i, j]
                        nlist.append(k)

                datan = pd.DataFrame(nlist).T
                colname = [val for val in range(784)]
                datan.columns = colname

                pixeldata = datan.values
                pixeldata = pixeldata / 255
                pixeldata = pixeldata.reshape(-1, 28, 28, 1)

                # Melakukan prediksi huruf menggunakan model
                prediction = model.predict(pixeldata)
                predarray = np.array(prediction[0])

                # Membuat dictionary hasil prediksi huruf beserta confidence
                letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
                predkey = max(letter_prediction_dict, key=letter_prediction_dict.get)
                predvalue = letter_prediction_dict[predkey]

                # Menampilkan hasil prediksi pada frame
                cv2.putText(frame, f"Predicted Character: {predkey} (Confidence: {100*predvalue:.2f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Menampilkan frame yang telah diproses
    cv2.imshow("Frame", frame)

# Melepaskan sumber daya video capture dan menutup jendela OpenCV setelah loop selesai
cap.release()
cv2.destroyAllWindows()
