import random

def from_binary(binary_str, num_bits=10):
    int_val = int(binary_str, 2)
    return int_val / (2 ** num_bits)

def pembentukan_populasi(populasi):
    # Konversi nilai biner ke skala asli
    lr_scaled = from_binary(populasi['LR'], 10)             # learning rate: antara 0.00001 - 0.01
    cnn_scaled = from_binary(populasi['CNN_Filter'], 10)    # jumlah filter CNN: 16 - 128
    lstm_scaled = from_binary(populasi['LSTM_Units'], 10)   # jumlah unit LSTM: 32 - 256
    seq_scaled = from_binary(populasi['Seq_Length'], 10)    # panjang sequence: 5 - 30
    dropout_scaled = from_binary(populasi['Dropout'], 10)   # dropout rate: 0.1 - 0.5
    
    # Normalisasi ke rentang [0, 1] agar bisa di-encode ke biner
    # Kembalikan ke rentang asli
    lr = 1e-5 + lr_scaled * (1e-2 - 1e-5)                    # learning rate asli
    cnn_filter = int(round(16 + cnn_scaled * (128 - 16)))   # normalisasi cnn_filter
    lstm_units = int(round(32 + lstm_scaled * (256 - 32)))  # normalisasi lstm units
    seq_len = int(round(5 + seq_scaled * (30 - 5)))         # normalisasi panjang urutan
    dropout = round(0.1 + dropout_scaled * (0.5 - 0.1), 3)   # normalisasi dropout
    #  lr_scaled = (lr - 1e-5) / (1e-2 - 1e-5)                        # normalisasi learning rate
    # cnn_scaled = (cnn_filter - 16) / (128 - 16)                   # normalisasi cnn_filter
    # lstm_scaled = (lstm_units - 32) / (256 - 32)                  # normalisasi lstm units
    # seq_scaled = (seq_len - 5) / (30 - 5)                         # normalisasi panjang urutan
    # dropout_scaled = (dropout - 0.1) / (0.5 - 0.1)                # normalisasi dropout
    populasi = {
        "num_conv_layers": 3,                   # default: 3 layer CNN
        "filters": [cnn_filter] * 3,            # 3 filter sama untuk tiap layer
        "kernel_sizes": [3, 3, 3],              # ukuran kernel CNN
        "activation": "relu",                   # fungsi aktivasi (default)
        "dropout_cnn": dropout,                 # dropout di bagian CNN

        "lstm_hidden_size": lstm_units,         # ukuran hidden state LSTM
        "lstm_num_layers": 2,                   # jumlah layer LSTM
        "bidirectional": True,                  # LSTM bidirectional
        "dropout_lstm": dropout,                # dropout di LSTM
        "seq_length": seq_len,                  # panjang urutan input

        "learning_rate": round(lr, 6),          # learning rate model
        "batch_size": 16,                       # batch size (default)
        "optimizer": "adam",                    # optimizer (default)
        "Fitness": populasi['Fitness']          # nilai fitness (tetap sama)
    }


    return populasi

