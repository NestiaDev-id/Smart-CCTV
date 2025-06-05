import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from ultralytics import YOLO
import cv2
import numpy as np
import os # Digunakan di contoh __main__
from collections import deque # Berguna untuk menyimpan sekuens fitur per objek

# Impor tracker Anda (pastikan path impornya benar)
# Asumsi tracker.py ada di ../yolo/ relatif terhadap file model ini jika struktur Anda adalah src/models/yolo_cnn_lstm_model.py
# dan src/models/yolo/track.py
try:
    from ..yolo.track import ObjectTracker # Sesuaikan path jika perlu
except ImportError:
    print("Peringatan: Tidak bisa mengimpor ObjectTracker. Tracking tidak akan berfungsi.")
    ObjectTracker = None # Fallback jika impor gagal

# --- Definisi Model CNN untuk Ekstraksi Fitur ---
def build_feature_extractor_cnn(input_shape_cnn, feature_vector_size=128):
    """
    Membangun model CNN yang dirancang khusus untuk ekstraksi fitur.
    Outputnya adalah feature vector.

    Args:
        input_shape_cnn (tuple): Bentuk input untuk CNN (height, width, channels).
        feature_vector_size (int): Ukuran feature vector yang diinginkan.

    Returns:
        tensorflow.keras.models.Model: Model CNN ekstraktor fitur.
    """
    cnn_input = Input(shape=input_shape_cnn, name="cnn_input_cropped_object")

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name="cnn_feat_conv1")(cnn_input)
    x = MaxPooling2D((2, 2), name="cnn_feat_pool1")(x)
    x = Dropout(0.25, name="cnn_feat_dropout1")(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="cnn_feat_conv2")(x)
    x = MaxPooling2D((2, 2), name="cnn_feat_pool2")(x)
    x = Dropout(0.25, name="cnn_feat_dropout2")(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name="cnn_feat_conv3")(x)
    x = MaxPooling2D((2, 2), name="cnn_feat_pool3")(x)
    # x = Dropout(0.25, name="cnn_feat_dropout3")(x) # Dropout sebelum flatten bisa opsional

    x = Flatten(name="cnn_feat_flatten")(x)
    cnn_feature_output = Dense(feature_vector_size, activation='relu', name="cnn_feat_vector")(x)
    
    feature_extractor_model = Model(inputs=cnn_input, outputs=cnn_feature_output, name="feature_extractor_cnn")
    print(f"Model CNN ekstraktor fitur dibangun dengan output dimensi: {feature_vector_size}")
    return feature_extractor_model

# --- Definisi Model LSTM untuk Klasifikasi Sekuens ---
def build_lstm_classifier(sequence_input_shape, num_classes, lstm_units=64, use_bidirectional=False):
    """
    Membangun model LSTM untuk klasifikasi sekuens fitur.

    Args:
        sequence_input_shape (tuple): Bentuk input sekuens (seq_length, feature_vector_dim).
        num_classes (int): Jumlah kelas output untuk klasifikasi perilaku/sekuens.
        lstm_units (int): Jumlah unit dalam layer LSTM.
        use_bidirectional (bool): Apakah menggunakan Bidirectional LSTM.

    Returns:
        tensorflow.keras.models.Model: Model LSTM classifier.
    """
    sequence_input = Input(shape=sequence_input_shape, name="lstm_input_feature_sequence")

    if use_bidirectional:
        x = Bidirectional(LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3, return_sequences=False), name="lstm_bi")(sequence_input)
        # recurrent_dropout mungkin lebih lambat di GPU, pertimbangkan untuk menghapusnya atau hanya menggunakan dropout
    else:
        x = LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3, return_sequences=False, name="lstm_uni")(sequence_input)

    x = Dense(64, activation='relu', name="lstm_dense_pre_output")(x)
    x = Dropout(0.5, name="lstm_dropout_dense")(x)
    lstm_output = Dense(num_classes, activation='softmax', name="lstm_output_classification")(x)

    lstm_model = Model(inputs=sequence_input, outputs=lstm_output, name="lstm_sequence_classifier")
    print(f"Model LSTM classifier dibangun untuk {num_classes} kelas.")
    return lstm_model


# --- Kelas Prosesor YOLO-CNN-LSTM ---
class YoloCnnLstmProcessor:
    def __init__(self, 
                 yolo_model_path="yolov8n.pt", 
                 cnn_input_size=(64, 64), # Ukuran lebih kecil mungkin lebih cepat untuk CNN per objek
                 cnn_feature_vector_size=128,
                 cnn_weights_path=None,
                 lstm_sequence_length=20, # Jumlah frame dalam satu sekuens untuk LSTM
                 lstm_units=64,
                 lstm_num_classes=3, # Contoh: 3 kelas perilaku (normal, aneh, berbahaya)
                 lstm_use_bidirectional=False,
                 lstm_weights_path=None,
                 target_yolo_classes=None,
                 yolo_confidence_threshold=0.5,
                 max_tracked_objects=20): # Batas jumlah objek yang dilacak untuk manajemen memori
        
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"Model YOLO '{yolo_model_path}' berhasil dimuat.")
        except Exception as e:
            print(f"Error saat memuat model YOLO '{yolo_model_path}': {e}")
            raise

        self.cnn_input_shape = (cnn_input_size[0], cnn_input_size[1], 3)
        self.cnn_feature_extractor = build_feature_extractor_cnn(self.cnn_input_shape, cnn_feature_vector_size)
        if cnn_weights_path:
            try:
                self.cnn_feature_extractor.load_weights(cnn_weights_path)
                print(f"Bobot CNN berhasil dimuat dari: {cnn_weights_path}")
            except Exception as e:
                print(f"Peringatan: Error memuat bobot CNN dari '{cnn_weights_path}': {e}.")
        
        self.lstm_sequence_length = lstm_sequence_length
        self.lstm_input_shape = (lstm_sequence_length, cnn_feature_vector_size)
        self.lstm_classifier = build_lstm_classifier(self.lstm_input_shape, lstm_num_classes, lstm_units, lstm_use_bidirectional)
        if lstm_weights_path:
            try:
                self.lstm_classifier.load_weights(lstm_weights_path)
                print(f"Bobot LSTM berhasil dimuat dari: {lstm_weights_path}")
            except Exception as e:
                print(f"Peringatan: Error memuat bobot LSTM dari '{lstm_weights_path}': {e}.")

        self.target_yolo_classes = target_yolo_classes if target_yolo_classes else []
        self.yolo_confidence_threshold = yolo_confidence_threshold

        if ObjectTracker:
            self.tracker = ObjectTracker()
            print("ObjectTracker berhasil diinisialisasi.")
        else:
            self.tracker = None
            print("ObjectTracker tidak tersedia.")
            
        # Penyimpanan sekuens fitur untuk setiap objek yang dilacak
        # Key: object_id, Value: deque (antrian) dari feature_vectors
        self.object_feature_sequences = {} 
        self.max_tracked_objects = max_tracked_objects

    def _preprocess_cropped_image_for_cnn(self, cropped_image):
        resized_image = cv2.resize(cropped_image, (self.cnn_input_shape[1], self.cnn_input_shape[0]))
        normalized_image = resized_image / 255.0
        return np.expand_dims(normalized_image, axis=0)

    def process_frame_for_yolo_cnn(self, frame):
        """ Hanya melakukan deteksi YOLO dan ekstraksi fitur CNN per objek. """
        yolo_results = self.yolo_model(frame, verbose=False)[0]
        
        current_frame_detections_for_tracker = [] # Untuk input ke tracker
        object_features_this_frame = {} # Key: bbox_tuple, Value: feature_vector

        for box in yolo_results.boxes:
            confidence = float(box.conf[0])
            if confidence >= self.yolo_confidence_threshold:
                class_id = int(box.cls[0])
                label_yolo = self.yolo_model.names[class_id]

                if not self.target_yolo_classes or label_yolo in self.target_yolo_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Data untuk tracker: [x1, y1, x2, y2, score, class_id]
                    current_frame_detections_for_tracker.append([x1, y1, x2, y2, confidence, class_id])
                    
                    if x1 < x2 and y1 < y2 and x1 >=0 and y1 >=0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                        cropped_object_img = frame[y1:y2, x1:x2]
                        if cropped_object_img.size == 0: continue
                        
                        cnn_input_data = self._preprocess_cropped_image_for_cnn(cropped_object_img)
                        feature_vector = self.cnn_feature_extractor.predict(cnn_input_data, verbose=0)[0]
                        object_features_this_frame[(x1,y1,x2,y2)] = feature_vector # Gunakan bbox sebagai key sementara
        
        return current_frame_detections_for_tracker, object_features_this_frame
    def process_single_live_frame(self, frame):
        # 1. Deteksi YOLO
        yolo_results = self.yolo_model(frame, verbose=False)[0]
        
        detections_for_tracker_input = []
        temp_features_for_this_frame = {} 

        for box in yolo_results.boxes:
            confidence = float(box.conf[0])
            if confidence >= self.yolo_confidence_threshold:
                class_id = int(box.cls[0])
                label_yolo = self.yolo_model.names[class_id]

                if not self.target_yolo_classes or label_yolo in self.target_yolo_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections_for_tracker_input.append([x1, y1, x2, y2, confidence, class_id])
                    
                    if x1 < x2 and y1 < y2 and x1 >=0 and y1 >=0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                        cropped_object_img = frame[y1:y2, x1:x2]
                        if cropped_object_img.size == 0: continue
                        
                        cnn_input_data = self._preprocess_cropped_image_for_cnn(cropped_object_img)
                        feature_vector = self.cnn_feature_extractor.predict(cnn_input_data, verbose=0)[0]
                        temp_features_for_this_frame[(x1,y1,x2,y2)] = {
                            "feature": feature_vector, 
                            "label_yolo": label_yolo, 
                            "confidence_yolo": confidence
                        }
        
        output_detections = []
        output_lstm_predictions = []

        if self.tracker and detections_for_tracker_input:
            tracked_objects = self.tracker.update(np.array(detections_for_tracker_input))
            active_ids_this_frame = set()

            for trk_obj in tracked_objects:
                tx1, ty1, tx2, ty2, obj_id_tracker = map(int, trk_obj[:5])
                active_ids_this_frame.add(obj_id_tracker)
                
                # --- Logika Pencocokan Fitur (Sederhana, perlu perbaikan untuk produksi) ---
                feature_data = temp_features_for_this_frame.get((tx1, ty1, tx2, ty2))

                if feature_data:
                    feature_vector = feature_data["feature"]
                    output_detections.append({
                        "id": obj_id_tracker, "bbox": [tx1, ty1, tx2, ty2],
                        "label_yolo": feature_data["label_yolo"], "confidence_yolo": feature_data["confidence_yolo"]
                    })

                    if obj_id_tracker not in self.object_feature_sequences:
                        if len(self.object_feature_sequences) >= self.max_tracked_objects:
                            try: oldest_id = next(iter(self.object_feature_sequences)); del self.object_feature_sequences[oldest_id]
                            except StopIteration: pass
                        self.object_feature_sequences[obj_id_tracker] = deque(maxlen=self.lstm_sequence_length)
                    
                    self.object_feature_sequences[obj_id_tracker].append(feature_vector)

                    if len(self.object_feature_sequences[obj_id_tracker]) == self.lstm_sequence_length:
                        current_sequence = np.array(self.object_feature_sequences[obj_id_tracker])
                        lstm_input_sequence = np.expand_dims(current_sequence, axis=0)
                        
                        lstm_prediction_probs = self.lstm_classifier.predict(lstm_input_sequence, verbose=0)[0]
                        predicted_class_lstm = np.argmax(lstm_prediction_probs)
                        
                        output_lstm_predictions.append({
                            "id": obj_id_tracker, "behavior_class_id": int(predicted_class_lstm),
                            "probabilities": lstm_prediction_probs.tolist()
                        })
            
            ids_to_remove = set(self.object_feature_sequences.keys()) - active_ids_this_frame
            for id_rem in ids_to_remove: del self.object_feature_sequences[id_rem]
        
        elif not self.tracker and temp_features_for_this_frame: 
            idx = 0
            for bbox_tuple, feature_data in temp_features_for_this_frame.items():
                output_detections.append({
                    "id": f"no_track_{idx}", "bbox": list(bbox_tuple),
                    "label_yolo": feature_data["label_yolo"], "confidence_yolo": feature_data["confidence_yolo"]
                })
                idx +=1

        return {
            "current_detections": output_detections,
            "new_lstm_predictions": output_lstm_predictions 
        }


    def process_video_sequence(self, video_frames_generator):
        """
        Memproses sekuens frame dari video, melakukan tracking, ekstraksi fitur,
        dan klasifikasi sekuens dengan LSTM.

        Args:
            video_frames_generator: Generator yang menghasilkan frame video satu per satu.
        
        Returns:
            dict: Key: object_id, Value: list prediksi LSTM untuk objek tersebut.
        """
        if not self.tracker:
            print("Error: Tracker tidak diinisialisasi. Tidak bisa memproses sekuens.")
            return {}

        all_lstm_predictions = {} # Untuk menyimpan semua prediksi LSTM per object_id

        frame_count = 0
        for frame in video_frames_generator:
            frame_count += 1
            print(f"Memproses frame ke-{frame_count}...")

            # 1. Deteksi YOLO dan Ekstraksi Fitur CNN
            detections_for_tracker, features_by_bbox = self.process_frame_for_yolo_cnn(frame)

            if not detections_for_tracker:
                # Update tracker dengan array kosong jika tidak ada deteksi
                self.tracker.update(np.empty((0, 6))) 
                continue

            # 2. Update Tracker
            # Tracker mengharapkan numpy array.
            tracked_objects = self.tracker.update(np.array(detections_for_tracker))
            # Output tracker: array of [x1, y1, x2, y2, object_id, class_id, (score jika tracker Anda menyediakannya)]

            current_frame_tracked_features = {} # Fitur untuk objek yang berhasil dilacak di frame ini

            for trk_obj in tracked_objects:
                tx1, ty1, tx2, ty2, obj_id = map(int, trk_obj[:5])
                # class_id_trk = int(trk_obj[5]) # Jika Anda butuh class_id dari tracker

                # Cari fitur yang sesuai dengan bbox yang dilacak
                # Ini mungkin perlu pencocokan IoU jika bbox dari YOLO dan tracker sedikit berbeda
                # Untuk kesederhanaan, kita cari bbox yang paling dekat/sama
                # Namun, lebih baik mengambil fitur berdasarkan deteksi awal sebelum tracking,
                # dan asosiasikan dengan ID setelah tracking.
                
                # Kita asumsikan `features_by_bbox` menggunakan bbox dari deteksi YOLO awal.
                # Kita perlu mencocokkan `obj_id` dari tracker dengan fitur yang diekstrak.
                # Ini memerlukan modifikasi: `process_frame_for_yolo_cnn` harus mengembalikan fitur
                # yang bisa dipetakan kembali ke deteksi sebelum tracking, atau `tracker.update`
                # harus bisa membantu asosiasi ini.

                # Pendekatan yang lebih baik:
                # a. `process_frame_for_yolo_cnn` mengekstrak fitur untuk semua deteksi YOLO yang valid.
                # b. `tracker.update` memberikan ID ke deteksi-deteksi tersebut.
                # c. Kita menggunakan ID ini untuk mengambil fitur yang sesuai.

                # Untuk saat ini, kita akan membuat asumsi sederhana bahwa bbox dari tracker
                # bisa digunakan untuk mengambil fitur (ini mungkin tidak selalu akurat).
                # Idealnya, asosiasi fitur dengan ID objek harus lebih robust.

                # Untuk sekarang, kita akan mengabaikan pencocokan bbox yang rumit dan fokus pada alur.
                # Kita asumsikan ada cara untuk mendapatkan fitur yang benar untuk obj_id.
                # Di implementasi nyata, Anda akan mengambil fitur dari `features_by_bbox`
                # berdasarkan bbox yang paling cocok dengan `(tx1, ty1, tx2, ty2)`.

                # Placeholder untuk mendapatkan fitur yang sesuai untuk obj_id
                # Mari kita ambil fitur pertama jika ada untuk demonstrasi (ini TIDAK BENAR untuk produksi)
                # Anda HARUS mengimplementasikan logika pencocokan fitur dengan ID objek yang benar.
                # Misalnya, simpan fitur bersamaan dengan deteksi awal, lalu setelah tracking,
                # gunakan ID untuk mengambil fitur yang sudah disimpan.

                # Untuk contoh ini, kita akan mengambil fitur acak dari yang terdeteksi di frame ini.
                # INI HANYA UNTUK ALUR DAN HARUS DIPERBAIKI.
                if features_by_bbox:
                    # Ambil fitur yang terkait dengan bbox (tx1,ty1,tx2,ty2) dari tracker
                    # Ini memerlukan pencocokan bbox dari yolo_results dengan hasil tracker.
                    # Untuk simplifikasi, kita coba cari bbox yang sama persis.
                    matched_feature = features_by_bbox.get((tx1,ty1,tx2,ty2))
                    if matched_feature is None:
                        # Cari yang paling dekat (IoU) jika tidak ada yang sama persis.
                        # Untuk saat ini, kita lewati jika tidak ada match persis.
                        # print(f"  Tidak ada fitur CNN yang cocok persis untuk bbox tracker: {(tx1,ty1,tx2,ty2)}")
                        continue
                else:
                    continue
                
                feature_vector = matched_feature

                # 3. Kelola Sekuens Fitur per Objek
                if obj_id not in self.object_feature_sequences:
                    if len(self.object_feature_sequences) >= self.max_tracked_objects:
                        # Hapus objek terlama jika batas tercapai (strategi sederhana)
                        oldest_id = next(iter(self.object_feature_sequences))
                        del self.object_feature_sequences[oldest_id]
                        if oldest_id in all_lstm_predictions: del all_lstm_predictions[oldest_id]
                        # print(f"Batas objek tercapai, menghapus data untuk ID: {oldest_id}")

                    self.object_feature_sequences[obj_id] = deque(maxlen=self.lstm_sequence_length)
                    if obj_id not in all_lstm_predictions:
                         all_lstm_predictions[obj_id] = []
                
                self.object_feature_sequences[obj_id].append(feature_vector)

                # 4. Jika sekuens cukup panjang, lakukan prediksi LSTM
                if len(self.object_feature_sequences[obj_id]) == self.lstm_sequence_length:
                    current_sequence = np.array(self.object_feature_sequences[obj_id])
                    # Tambahkan dimensi batch (LSTM mengharapkan input batch)
                    lstm_input_sequence = np.expand_dims(current_sequence, axis=0)
                    
                    lstm_prediction_probs = self.lstm_classifier.predict(lstm_input_sequence, verbose=0)[0]
                    predicted_class_lstm = np.argmax(lstm_prediction_probs)
                    
                    all_lstm_predictions[obj_id].append({
                        'frame': frame_count,
                        'class_probs': lstm_prediction_probs.tolist(),
                        'predicted_class': predicted_class_lstm
                    })
                    print(f"  Prediksi LSTM untuk Objek ID {obj_id} di Frame {frame_count}: Kelas {predicted_class_lstm} (Probs: {lstm_prediction_probs})")

            # Bersihkan objek yang sudah tidak dilacak dari dictionary sekuens
            active_ids_this_frame = {int(trk[4]) for trk in tracked_objects}
            ids_to_remove = set(self.object_feature_sequences.keys()) - active_ids_this_frame
            for id_rem in ids_to_remove:
                del self.object_feature_sequences[id_rem]
                # print(f"Objek ID {id_rem} tidak lagi dilacak, menghapus sekuens fiturnya.")
        
        print("Pemrosesan video selesai.")
        return all_lstm_predictions


# --- Contoh Penggunaan ---
if __name__ == "__main__":
    # --- Konfigurasi ---
    YOLO_MODEL = "yolov8n.pt" # Ini akan diunduh jika belum ada di direktori kerja
    CNN_WEIGHTS = None 
    LSTM_WEIGHTS = None 
    
    CNN_INPUT_SIZE_HW = (64, 64)
    CNN_FEATURE_DIM = 128
    
    LSTM_SEQ_LEN = 15 
    LSTM_UNITS = 64
    # Definisikan nama kelas perilaku Anda di sini agar bisa ditampilkan
    BEHAVIOR_CLASS_NAMES = { 
        0: "Normal", 
        1: "Aneh", 
        2: "Berbahaya" 
    } # Sesuaikan dengan jumlah dan nama kelas Anda
    NUM_BEHAVIOR_CLASSES = len(BEHAVIOR_CLASS_NAMES)
    USE_BIDIRECTIONAL_LSTM = False

    TARGET_CLASSES_FROM_YOLO = ['person', 'car', 'truck', 'bus', 'motorbike']
    YOLO_CONF_THRESHOLD = 0.45

    # --- Path Video ---
    # Tentukan path ke video input Anda
    VIDEO_INPUT_PATH = os.path.join(
        os.path.dirname(__file__), # Direktori saat ini (src/models/yolo)
        "..", # Naik ke src/models
        "..", # Naik ke src
        "..", # Naik ke root proyek (D:/Earth/Smart-CCTV)
        "data", "raw", "video30.mp4"
    )
    # Path untuk menyimpan video output
    VIDEO_OUTPUT_PATH = os.path.join(
        os.path.dirname(__file__), 
        "..", "..", "..",
        "data", "processed", "video30_processed_output.mp4" # Simpan di data/processed
    )
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed"), exist_ok=True)


    # Inisialisasi Prosesor
    print("Menginisialisasi YoloCnnLstmProcessor...")
    processor = YoloCnnLstmProcessor(
        yolo_model_path=YOLO_MODEL,
        cnn_input_size=CNN_INPUT_SIZE_HW,
        cnn_feature_vector_size=CNN_FEATURE_DIM,
        cnn_weights_path=CNN_WEIGHTS,
        lstm_sequence_length=LSTM_SEQ_LEN,
        lstm_units=LSTM_UNITS,
        lstm_num_classes=NUM_BEHAVIOR_CLASSES,
        lstm_use_bidirectional=USE_BIDIRECTIONAL_LSTM,
        lstm_weights_path=LSTM_WEIGHTS,
        target_yolo_classes=TARGET_CLASSES_FROM_YOLO,
        yolo_confidence_threshold=YOLO_CONF_THRESHOLD
    )
    print("Inisialisasi Selesai.")

    # --- Membaca Video Input ---
    if not os.path.exists(VIDEO_INPUT_PATH):
        print(f"ERROR: File video input tidak ditemukan di {VIDEO_INPUT_PATH}")
        exit()

    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka video {VIDEO_INPUT_PATH}")
        exit()

    # Dapatkan properti video untuk VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Input: {VIDEO_INPUT_PATH} ({frame_width}x{frame_height} @ {fps:.2f} FPS)")

    # --- Inisialisasi Video Output ---
    # Gunakan codec 'mp4v' untuk .mp4 atau 'XVID' untuk .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Video Output akan disimpan ke: {VIDEO_OUTPUT_PATH}")

    frame_count = 0
    # Dictionary untuk menyimpan prediksi LSTM terakhir untuk setiap ID objek yang aktif
    last_lstm_behavior_for_id = {}

    print("\nMemulai pemrosesan video frame per frame...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Selesai membaca semua frame atau error saat membaca frame.")
            break

        frame_count += 1
        if frame_count % 10 == 0: # Print status setiap 10 frame
             print(f"  Memproses frame ke-{frame_count}...")

        output_frame_to_draw = frame.copy() # Buat salinan frame untuk digambari

        # Proses frame menggunakan YoloCnnLstmProcessor
        # Metode ini harus mengembalikan deteksi saat ini dan prediksi LSTM baru
        results_per_frame = processor.process_single_live_frame(frame)
        
        current_detections = results_per_frame.get('current_detections', [])
        new_lstm_predictions = results_per_frame.get('new_lstm_predictions', [])

        # Update prediksi LSTM terakhir untuk ID objek
        for lstm_pred in new_lstm_predictions:
            obj_id = lstm_pred.get('id')
            behavior_id = lstm_pred.get('behavior_class_id')
            if obj_id is not None and behavior_id is not None:
                last_lstm_behavior_for_id[obj_id] = BEHAVIOR_CLASS_NAMES.get(behavior_id, f"Kelas {behavior_id}")

        # Gambar hasil deteksi dan prediksi pada output_frame_to_draw
        for det in current_detections:
            obj_id = det.get('id')
            bbox = det.get('bbox')
            label_yolo = det.get('label_yolo', 'N/A')
            confidence_yolo = det.get('confidence_yolo', 0.0)

            if bbox:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) # Hijau untuk bounding box YOLO
                cv2.rectangle(output_frame_to_draw, (x1, y1), (x2, y2), color, 2)
                
                # Label YOLO dan ID Objek
                yolo_text = f"{label_yolo} ({confidence_yolo:.2f})"
                if obj_id is not None:
                    yolo_text += f" ID:{obj_id}"
                
                text_y_pos = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(output_frame_to_draw, yolo_text, (x1, text_y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Tampilkan prediksi perilaku LSTM jika ada untuk ID objek ini
                if obj_id is not None and obj_id in last_lstm_behavior_for_id:
                    behavior_text = f"P: {last_lstm_behavior_for_id[obj_id]}" # P: Perilaku
                    cv2.putText(output_frame_to_draw, behavior_text, (x1, y2 + 20), # Di bawah bbox
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2) # Warna biru-cyan

        # Tulis frame yang sudah dianotasi ke file video output
        out_video_writer.write(output_frame_to_draw)

        # (Opsional) Tampilkan frame yang sedang diproses
        # cv2.imshow("Processed Frame", output_frame_to_draw)
        # if cv2.waitKey(1) & 0xFF == ord('q'): # Tekan 'q' untuk keluar
        #     break
            
    # Setelah loop selesai, lepaskan semua resource
    cap.release()
    out_video_writer.release()
    # cv2.destroyAllWindows() # Jika menggunakan cv2.imshow()

    print(f"\nPemrosesan video selesai. Output disimpan di: {VIDEO_OUTPUT_PATH}")
    print("Contoh visualisasi selesai.")