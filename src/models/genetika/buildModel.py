from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD

def build_model(params):
    input_shape = (params['seq_length'], 64, 64, 3)  # misalnya input adalah frame video (sequence of images)
    input_layer = Input(shape=input_shape)

    # Reshape agar jadi batch*seq_length, 64, 64, 3
    reshaped = Reshape((input_shape[0] * input_shape[1], input_shape[2], input_shape[3]))(input_layer)

    # CNN path (gunakan TimeDistributed kalau input 3D)
    x = reshaped
    for i in range(params['num_conv_layers']):
        x = Conv2D(
            filters=params['filters'][i],
            kernel_size=(params['kernel_sizes'][i], params['kernel_sizes'][i]),
            activation=params['activation'],
            padding='same'
        )(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    
    # Reshape ke (batch_size, timesteps, features)
    x = Reshape((params['seq_length'], -1))(x)

    # LSTM layer
    if params['bidirectional']:
        x = Bidirectional(LSTM(
            params['lstm_hidden_size'],
            return_sequences=False,
            dropout=params['dropout_lstm'],
            recurrent_dropout=0.1
        ))(x)
    else:
        x = LSTM(
            params['lstm_hidden_size'],
            return_sequences=False,
            dropout=params['dropout_lstm'],
            recurrent_dropout=0.1
        )(x)

    # Dense + Dropout
    x = Dropout(params['dropout_lstm'])(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # misalnya klasifikasi 10 kelas

    model = Model(inputs=input_layer, outputs=output)

    # Optimizer sesuai parameter
    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=params['learning_rate'])
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
