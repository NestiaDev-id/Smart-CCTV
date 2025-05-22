import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# Anda mungkin perlu callback seperti EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping 

import numpy as np # Berguna untuk data dummy jika diperlukan untuk testing awal
import random # Sudah ada di genetika.py

# --- (Definisi HYPERPARAMETER_RANGES dan fungsi encode/decode sudah ada di genetika.py) ---

def build_cnn_lstm_model(input_shape, num_classes, hyperparams):
    """
    Membangun arsitektur model CNN-LSTM berdasarkan hyperparameter yang diberikan.

    Args:
        input_shape (tuple): Bentuk data input per sekuens 
                             (seq_length, height, width, channels).
        num_classes (int): Jumlah kelas output untuk klasifikasi.
        hyperparams (dict): Dictionary berisi hyperparameter yang sudah di-decode.
                            Contoh: {'learning_rate': 0.001, 'cnn_filters_l1': 32, ...}

    Returns:
        tensorflow.keras.models.Model: Model CNN-LSTM yang belum di-compile.
    """
    
    # --- Validasi Hyperparameter (Contoh Sederhana) ---
    # Pastikan hyperparameter yang dibutuhkan ada
    required_cnn_params = ['cnn_filters_l1', 'cnn_kernel_l1', 'activation_cnn', 'dropout_cnn']
    required_lstm_params = ['lstm_units']
    
    for p in required_cnn_params + required_lstm_params:
        if p not in hyperparams:
            raise ValueError(f"Hyperparameter '{p}' tidak ditemukan.")

    # --- Input Layer ---
    # Inputnya adalah sekuens gambar, jadi kita gunakan TimeDistributed untuk CNN
    # atau kita bisa merancang CNN untuk memproses sekuens secara langsung jika menggunakan Conv3D.
    # Untuk TimeDistributed CNN, input_shape_cnn adalah (height, width, channels)
    # dan input_layer akan memiliki shape (seq_length, height, width, channels)
    
    # Contoh: jika hyperparams['seq_length'] adalah bagian dari optimasi GA
    # dan input_shape yang di-pass ke fungsi ini sudah (None, height, width, channels) untuk satu frame.
    # Maka kita perlu menyesuaikan input_shape di sini.
    # Untuk saat ini, asumsikan input_shape sudah benar (seq_length, height, width, channels).
    
    img_input = Input(shape=input_shape, name="input_frames_sequence")

    # --- CNN Layers (menggunakan TimeDistributed) ---
    # TimeDistributed menerapkan layer yang sama ke setiap item temporal (setiap frame dalam sekuens).
    x = img_input
    
    # Layer CNN 1
    x = TimeDistributed(
        Conv2D(
            filters=hyperparams['cnn_filters_l1'],
            kernel_size=(hyperparams['cnn_kernel_l1'], hyperparams['cnn_kernel_l1']),
            padding='same',
            activation=hyperparams['activation_cnn']
        ), name="cnn_conv1_td"
    )(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name="cnn_pool1_td")(x)
    if hyperparams.get('dropout_cnn', 0) > 0: # Jika dropout_cnn didefinisikan dan > 0
        x = TimeDistributed(Dropout(hyperparams['dropout_cnn']), name="cnn_dropout1_td")(x)

    # (Opsional) Tambahkan lebih banyak layer CNN berdasarkan hyperparameter
    # num_cnn_layers = hyperparams.get('num_cnn_layers', 1)
    # if num_cnn_layers > 1 and 'cnn_filters_l2' in hyperparams and 'cnn_kernel_l2' in hyperparams:
    #     x = TimeDistributed(
    #         Conv2D(
    #             filters=hyperparams['cnn_filters_l2'],
    #             kernel_size=(hyperparams['cnn_kernel_l2'], hyperparams['cnn_kernel_l2']),
    #             padding='same',
    #             activation=hyperparams['activation_cnn']
    #         ), name="cnn_conv2_td"
    #     )(x)
    #     x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name="cnn_pool2_td")(x)
    #     if hyperparams.get('dropout_cnn', 0) > 0:
    #         x = TimeDistributed(Dropout(hyperparams['dropout_cnn']), name="cnn_dropout2_td")(x)
            
    # Flatten output dari CNN (masih dalam TimeDistributed)
    # Output CNN per frame perlu di-flatten sebelum masuk ke LSTM
    x = TimeDistributed(Flatten(), name="cnn_flatten_td")(x)

    # --- LSTM Layers ---
    # Parameter yang mungkin: lstm_units, dropout_lstm, use_bidirectional
    use_bidirectional = hyperparams.get('use_bidirectional', False) # Ambil dari hyperparams jika ada
    dropout_lstm_rate = hyperparams.get('dropout_lstm', 0.2) # Contoh default

    if use_bidirectional:
        lstm_layer = Bidirectional(
            LSTM(
                units=hyperparams['lstm_units'],
                dropout=dropout_lstm_rate, # Dropout pada input/recurrent connections
                # recurrent_dropout=dropout_lstm_rate, # recurrent_dropout seringkali lebih lambat di GPU
                return_sequences=False # False karena kita ingin output akhir setelah memproses seluruh sekuens
                                       # Jika ada layer LSTM lagi, ini harus True untuk layer sebelumnya
            ), name="lstm_bidirectional"
        )
    else:
        lstm_layer = LSTM(
            units=hyperparams['lstm_units'],
            dropout=dropout_lstm_rate,
            # recurrent_dropout=dropout_lstm_rate,
            return_sequences=False,
            name="lstm_unidirectional"
        )
    
    x = lstm_layer(x) # LSTM memproses sekuens fitur dari TimeDistributed(Flatten())

    # --- Dense Layers (Fully Connected) ---
    # Parameter yang mungkin: dense_units, activation_dense, dropout_dense
    dense_units = hyperparams.get('dense_units', 64) # Contoh default
    activation_dense = hyperparams.get('activation_dense', 'relu') # Contoh default
    dropout_dense_rate = hyperparams.get('dropout_dense', 0.5) # Contoh default

    if dense_units > 0 : # Hanya tambah dense layer jika unitnya > 0
        x = Dense(units=dense_units, activation=activation_dense, name="dense_layer1")(x)
        if dropout_dense_rate > 0:
            x = Dropout(dropout_dense_rate, name="dense_dropout1")(x)
            
    # Output Layer
    # Aktivasi 'softmax' untuk klasifikasi multi-kelas
    # Aktivasi 'sigmoid' untuk klasifikasi biner atau multi-label
    output_activation = 'softmax' if num_classes > 1 else 'sigmoid' # Asumsi >1 kelas = multiclass
    if num_classes == 1: output_activation = 'sigmoid' # untuk biner

    output_layer = Dense(num_classes, activation=output_activation, name="output_classification")(x)

    # --- Buat Model ---
    model = Model(inputs=img_input, outputs=output_layer)
    
    return model

def compile_and_train_model(model, hyperparams, train_X, train_y, val_X, val_y, epochs_ga):
    """
    Meng-compile dan melatih model yang sudah dibangun.
    """
    # --- Pilih Optimizer ---
    optimizer_type = hyperparams.get('optimizer', 'adam').lower()
    learning_rate = hyperparams.get('learning_rate', 0.001)

    if optimizer_type == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9) # Momentum seringkali membantu SGD
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer tidak dikenal: {optimizer_type}")

    # --- Compile Model ---
    # Tentukan loss function berdasarkan jumlah kelas
    if model.output_shape[-1] == 1: # Output layer memiliki 1 unit (biner)
        loss_function = 'binary_crossentropy'
    else: # Output layer > 1 unit (multi-kelas)
        loss_function = 'categorical_crossentropy' 
        # Pastikan y_train dan y_val di-encode sebagai one-hot jika menggunakan categorical_crossentropy

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    # model.summary() # Bisa berguna untuk debugging, tapi mungkin terlalu verbose untuk GA

    # --- Callback (Opsional tapi Direkomendasikan) ---
    # Early stopping untuk menghentikan training jika tidak ada peningkatan pada validation loss
    early_stopping = EarlyStopping(
        monitor='val_loss',    # Metrik yang dipantau
        patience=5,            # Jumlah epoch tanpa peningkatan sebelum berhenti
        verbose=0,             # 0 = silent, 1 = print saat berhenti
        restore_best_weights=True # Kembalikan bobot terbaik saat berhenti
    )

    # --- Train Model ---
    batch_size = hyperparams.get('batch_size', 32) # Ambil batch_size dari hyperparams jika ada

    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=epochs_ga,
        batch_size=batch_size,
        callbacks=[early_stopping], # Tambahkan callback di sini
        verbose=0 # 0 = silent, 1 = progress bar, 2 = one line per epoch. Untuk GA, 0 biasanya terbaik.
    )
    
    return history

def build_and_train_model_for_ga(hyperparams, input_shape_per_frame, num_classes,
                                 train_X_seq, train_y_seq, 
                                 val_X_seq, val_y_seq, 
                                 epochs_ga=10):
    """
    Fungsi utama untuk Algoritma Genetika:
    1. Membangun model CNN-LSTM berdasarkan hyperparams.
    2. Meng-compile model.
    3. Melatih model pada train_X, train_y.
    4. Mengevaluasi pada val_X, val_y.
    5. Mengembalikan metrik fitness (misalnya, validation accuracy).

    Args:
        hyperparams (dict): Dictionary hyperparameter.
        input_shape_per_frame (tuple): Bentuk satu frame (height, width, channels).
        num_classes (int): Jumlah kelas output.
        train_X_seq (np.array): Data latih, shape (num_samples, seq_length, height, width, channels).
        train_y_seq (np.array): Label latih, shape (num_samples, num_classes) untuk categorical
                               atau (num_samples, 1) untuk biner.
        val_X_seq (np.array): Data validasi.
        val_y_seq (np.array): Label validasi.
        epochs_ga (int): Jumlah epoch untuk melatih setiap model dalam GA.

    Returns:
        float: Nilai fitness (misalnya, akurasi validasi tertinggi atau loss validasi terendah).
               Jika akurasi, kembalikan nilai positif (lebih tinggi lebih baik).
               Jika loss, kembalikan nilai negatif dari loss atau 1/loss (agar lebih tinggi lebih baik).
               Atau, modifikasi GA untuk meminimalkan fitness.
    """
    print(f"  Mengevaluasi hyperparams: {hyperparams}")
    try:
        # Dapatkan seq_length dari hyperparameter atau dari data jika tetap
        seq_length = hyperparams.get('seq_length', train_X_seq.shape[1] if train_X_seq is not None else 10)
        
        # Bentuk input untuk model CNN-LSTM
        # (seq_length, height, width, channels)
        model_input_shape = (seq_length, input_shape_per_frame[0], input_shape_per_frame[1], input_shape_per_frame[2])

        # Jika seq_length dari GA berbeda dengan data yang ada, perlu ada penyesuaian data
        # Untuk sekarang, asumsikan train_X_seq sudah memiliki seq_length yang sesuai atau
        # kita akan memotong/padding data di sini.
        # Paling sederhana adalah jika train_X_seq sudah memiliki seq_length yang dinamis.
        # Jika tidak, Anda perlu fungsi untuk menyesuaikan train_X_seq dan val_X_seq ke seq_length dari hyperparams.

        # --- (1) Bangun Model ---
        model = build_cnn_lstm_model(model_input_shape, num_classes, hyperparams)
        
        # --- (2) & (3) Compile dan Latih Model ---
        history = compile_and_train_model(model, hyperparams, 
                                          train_X_seq, train_y_seq, 
                                          val_X_seq, val_y_seq, 
                                          epochs_ga)
        
        # --- (4) Dapatkan Metrik Fitness ---
        # Ambil akurasi validasi terbaik dari history (karena ada EarlyStopping restore_best_weights)
        # atau akurasi validasi terakhir jika tidak pakai restore_best_weights.
        if 'val_accuracy' in history.history and history.history['val_accuracy']:
            # Jika menggunakan restore_best_weights, val_accuracy terbaik ada di epoch sebelum berhenti
            # Namun, cara termudah adalah mengambil nilai maksimum dari history.
            fitness_value = max(history.history['val_accuracy'])
        else: # Jika tidak ada val_accuracy (misal, training gagal atau sangat singkat)
            fitness_value = 0.0 # Fitness buruk
            
        print(f"  Selesai evaluasi. Akurasi Validasi: {fitness_value:.4f}")
        
        # Bersihkan session Keras untuk mencegah kebocoran memori (penting untuk GA)
        tf.keras.backend.clear_session()
        del model
        del history
        
        return fitness_value

    except Exception as e:
        print(f"  ERROR saat membangun/melatih model untuk GA: {e}")
        print(f"  Hyperparameters yang menyebabkan error: {hyperparams}")
        tf.keras.backend.clear_session() # Bersihkan juga jika error
        return float('-inf') # Fitness sangat buruk jika ada error