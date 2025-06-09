# models/cnn_model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def build_feature_extractor_cnn(input_shape_cnn, feature_vector_size=128, activation='relu', filters=32, kernel_size=3, dropout_rate=0.25):
    """
    Membangun model CNN yang dirancang khusus untuk ekstraksi fitur.
    Outputnya adalah feature vector.

    Args:
        input_shape_cnn (tuple): Bentuk input untuk CNN (height, width, channels).
        feature_vector_size (int): Ukuran feature vector yang diinginkan.
        activation (str): Fungsi aktivasi untuk layer konvolusi.
        filters (int): Jumlah filter di layer konvolusi pertama.
        kernel_size (int): Ukuran kernel untuk layer konvolusi.
        dropout_rate (float): Tingkat dropout.

    Returns:
        tensorflow.keras.models.Model: Model CNN ekstraktor fitur.
    """
    cnn_input = Input(shape=input_shape_cnn, name="cnn_input_cropped_object")

    x = Conv2D(filters, (kernel_size, kernel_size), activation=activation, padding='same', name="cnn_feat_conv1")(cnn_input)
    x = MaxPooling2D((2, 2), name="cnn_feat_pool1")(x)
    x = Dropout(dropout_rate, name="cnn_feat_dropout1")(x)

    x = Conv2D(filters * 2, (kernel_size, kernel_size), activation=activation, padding='same', name="cnn_feat_conv2")(x)
    x = MaxPooling2D((2, 2), name="cnn_feat_pool2")(x)
    x = Dropout(dropout_rate, name="cnn_feat_dropout2")(x)
    
    x = Flatten(name="cnn_feat_flatten")(x)
    cnn_feature_output = Dense(feature_vector_size, activation=activation, name="cnn_feat_vector")(x)
    
    feature_extractor_model = Model(inputs=cnn_input, outputs=cnn_feature_output, name="feature_extractor_cnn")
    print(f"Model CNN ekstraktor fitur dibangun dengan output dimensi: {feature_vector_size}")
    return feature_extractor_model