from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.optimizers import Adam, SGD

def build_yolo_cnn_lstm_model(params):
    # Input: sekuens gambar (misalnya, hasil crop objek dari YOLO)
    # params['seq_length'] adalah jumlah frame dalam satu sekuens
    # params['img_height'], params['img_width'], params['img_channels'] adalah dimensi gambar objek
    input_shape = (params['seq_length'], params['img_height'], params['img_width'], params['img_channels'])
    input_layer = Input(shape=input_shape)

    # CNN path (menggunakan TimeDistributed untuk menerapkan CNN pada setiap frame dalam sekuens)
    cnn_model = Conv2D(filters=params['filters'][0], kernel_size=(params['kernel_sizes'][0], params['kernel_sizes'][0]), activation=params['activation'], padding='same')(input_layer)
    cnn_model = MaxPooling2D(pool_size=(2, 2))(cnn_model)

    for i in range(1, params['num_conv_layers']):
        cnn_model = Conv2D(filters=params['filters'][i], kernel_size=(params['kernel_sizes'][i], params['kernel_sizes'][i]), activation=params['activation'], padding='same')(cnn_model)
        cnn_model = MaxPooling2D(pool_size=(2, 2))(cnn_model)

    # Menggunakan TimeDistributed untuk Flatten dan Dense layer jika fitur diekstrak per frame
    # sebelum masuk ke LSTM
    # Jika CNN mengekstrak fitur dari seluruh sekuens sekaligus (misalnya, menggunakan Conv3D),
    # maka TimeDistributed tidak diperlukan di sini.
    # Asumsi di sini adalah kita menerapkan CNN ke setiap frame secara independen.
    
    # Untuk menerapkan Flatten ke setiap frame dalam sekuens
    # Output dari CNN terakhir (sebelum flatten) adalah (None, seq_length, h_cnn, w_cnn, num_filters_cnn)
    # Kita perlu mereshape atau menggunakan TimeDistributed(Flatten())
    
    # Contoh dengan TimeDistributed(Flatten())
    # Kemudian bisa diikuti TimeDistributed(Dense(...)) untuk mendapatkan feature vector per frame
    time_distributed_flatten = TimeDistributed(Flatten())(cnn_model)
    
    # (Optional) Dense layer setelah Flatten per frame
    # time_distributed_dense = TimeDistributed(Dense(params['dense_units_after_cnn'], activation=params['activation']))(time_distributed_flatten)
    # lstm_input = time_distributed_dense
    lstm_input = time_distributed_flatten # Jika tidak ada Dense layer setelah Flatten

    # LSTM layer
    if params['bidirectional']:
        lstm_output = Bidirectional(LSTM(
            params['lstm_hidden_size'],
            return_sequences=False, # False karena kita ingin output akhir setelah memproses seluruh sekuens
            dropout=params['dropout_lstm']
            # recurrent_dropout tidak lagi direkomendasikan untuk GPU di Keras versi baru, 
            # pertimbangkan alternatif atau hapus jika ada isu performa/kompatibilitas
        ))(lstm_input)
    else:
        lstm_output = LSTM(
            params['lstm_hidden_size'],
            return_sequences=False,
            dropout=params['dropout_lstm']
        )(lstm_input)

    # Dense + Dropout
    x = Dropout(params['dropout_dense_after_lstm'])(lstm_output) # Dropout setelah LSTM
    x = Dense(params['dense_layer_size'], activation=params['activation'])(x) # Dense layer setelah LSTM
    output = Dense(params['num_classes'], activation='softmax')(x) # Output layer (misalnya klasifikasi)

    model = Model(inputs=input_layer, outputs=output)

    # Optimizer sesuai parameter
    if params['optimizer'].lower() == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'].lower() == 'sgd':
        optimizer = SGD(learning_rate=params['learning_rate'])
    else:
        raise ValueError("Optimizer tidak didukung")

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model