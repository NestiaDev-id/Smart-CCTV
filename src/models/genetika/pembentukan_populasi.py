import random

def to_binary(value, num_bits=10):
    scaled_value = int(value * (2 ** num_bits))
    return bin(scaled_value)[2:].zfill(num_bits)

def pembentukan_populasi_awal(jumlah_kromosom):
    populasi = []

    for _ in range(jumlah_kromosom):
        # Generate nilai acak dalam range yang telah ditentukan
        lr = random.uniform(1e-5, 1e-2)  # learning rate
        cnn_filter = random.randint(16, 128)  # filter CNN
        lstm_units = random.randint(32, 256)  # unit LSTM
        seq_len = random.randint(5, 30)  # panjang urutan
        dropout = random.uniform(0.1, 0.5)  # dropout rate

        # Skala ke [0, 1] untuk encode ke biner
        lr_scaled = (lr - 1e-5) / (1e-2 - 1e-5)
        cnn_scaled = (cnn_filter - 16) / (128 - 16)
        lstm_scaled = (lstm_units - 32) / (256 - 32)
        seq_scaled = (seq_len - 5) / (30 - 5)
        dropout_scaled = (dropout - 0.1) / (0.5 - 0.1)

        # Encode ke biner
        individu = {
            'LR': to_binary(lr_scaled),
            'CNN_Filter': to_binary(cnn_scaled),
            'LSTM_Units': to_binary(lstm_scaled),
            'Seq_Length': to_binary(seq_scaled),
            'Dropout': to_binary(dropout_scaled),
            'Fitness': None
        }

        populasi.append(individu)

    return populasi
