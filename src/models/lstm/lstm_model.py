# models/lstm_model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model

def build_lstm_classifier(sequence_input_shape, num_classes, lstm_units=64, use_bidirectional=False, dropout_rate=0.5):
    """
    Membangun model LSTM untuk klasifikasi sekuens fitur.

    Args:
        sequence_input_shape (tuple): Bentuk input sekuens (seq_length, feature_vector_dim).
        num_classes (int): Jumlah kelas output untuk klasifikasi.
        lstm_units (int): Jumlah unit dalam layer LSTM.
        use_bidirectional (bool): Apakah menggunakan Bidirectional LSTM.
        dropout_rate (float): Tingkat dropout.

    Returns:
        tensorflow.keras.models.Model: Model LSTM classifier.
    """
    sequence_input = Input(shape=sequence_input_shape, name="lstm_input_feature_sequence")

    lstm_layer = LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3, return_sequences=False, name="lstm_layer")
    if use_bidirectional:
        x = Bidirectional(lstm_layer, name="lstm_bi")(sequence_input)
    else:
        x = lstm_layer(sequence_input)

    x = Dense(lstm_units // 2, activation='relu', name="lstm_dense_pre_output")(x)
    x = Dropout(dropout_rate, name="lstm_dropout_dense")(x)
    lstm_output = Dense(num_classes, activation='softmax', name="lstm_output_classification")(x)

    lstm_model = Model(inputs=sequence_input, outputs=lstm_output, name="lstm_sequence_classifier")
    print(f"Model LSTM classifier dibangun untuk {num_classes} kelas.")
    return lstm_model