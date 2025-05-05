def evaluasi_fitness(populasi, train, test):
    for individu in populasi:
        lr = decode_lr(to_decimal(individu['LR']))
        cnn_filter = int(to_decimal(individu['CNN_Filter']) * (128 - 16) + 16)
        lstm_units = int(to_decimal(individu['LSTM_Units']) * (256 - 32) + 32)
        seq_len = int(to_decimal(individu['Seq_Length']) * (30 - 5) + 5)
        dropout = to_decimal(individu['Dropout']) * 0.5
        
        # Model training
        model = build_cnn_lstm_model(lr, cnn_filter, lstm_units, dropout, seq_len)
        history = model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=1, verbose=0)
        
        # Fitness = validation loss
        individu['Fitness'] = history.history['val_loss'][-1]
    return populasi
