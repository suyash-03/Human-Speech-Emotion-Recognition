import numpy as np # linear algebra
import pandas as pd
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display

import warnings
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint


def create_LSTM() :
    model = Sequential([
        LSTM(256, return_sequences=False, input_shape=(40,1)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])
    return model

if __name__ == "__main__": 
    warnings.filterwarnings('ignore')

    paths = []
    labels = []
    for dirname, _, filenames in os.walk('./TESS Toronto emotional speech set data'):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
        if len(paths) == 2800:
            break
    print('Dataset is Loaded')

    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels
    df.head()

    df['label'].value_counts()



    emotion = 'fear'
    path = np.array(df['speech'][df['label']==emotion])[0]
    data, sampling_rate = librosa.load(path)
    Audio(path)

    emotion = 'angry'
    path = np.array(df['speech'][df['label']==emotion])[1]
    data, sampling_rate = librosa.load(path)
    Audio(path)

    emotion = 'disgust'
    path = np.array(df['speech'][df['label']==emotion])[0]
    data, sampling_rate = librosa.load(path)
    Audio(path)

    emotion = 'neutral'
    path = np.array(df['speech'][df['label']==emotion])[0]
    data, sampling_rate = librosa.load(path)
    Audio(path)

    emotion = 'sad'
    path = np.array(df['speech'][df['label']==emotion])[0]
    data, sampling_rate = librosa.load(path)
    Audio(path)

    emotion = 'ps'
    path = np.array(df['speech'][df['label']==emotion])[0]
    data, sampling_rate = librosa.load(path)
    Audio(path)

    emotion = 'happy'
    path = np.array(df['speech'][df['label']==emotion])[0]
    data, sampling_rate = librosa.load(path)
    Audio(path)

    def extract_mfcc(filename):
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc

    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
    X = [x for x in X_mfcc]
    X = np.array(X)
    X.shape
    X = np.expand_dims(X, -1)
    X.shape

    from sklearn.preprocessing import OneHotEncoder   # classification of labels
    enc = OneHotEncoder()
    y = enc.fit_transform(df[['label']])
    y = y.toarray()

    model = create_LSTM()

    model.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Update the same weights file, everytime val accuracy is beat
    checkpoint = ModelCheckpoint('./saved_weights/model_best.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64, callbacks=[checkpoint])

    # Select a sample from the dataset for prediction
    sample_index = 0  # Change this index to select a different sample
    sample_path = df['speech'][sample_index]
    sample_label = df['label'][sample_index]

    # Load and extract MFCC features for the selected sample
    sample_data, _ = librosa.load(sample_path, duration=3, offset=0.5)
    sample_mfcc = extract_mfcc(sample_path)
    sample_mfcc = np.expand_dims(sample_mfcc, 0)
    sample_mfcc = np.expand_dims(sample_mfcc, -1)

    # Predict the emotion label using the trained model
    predicted_probabilities = model.predict(sample_mfcc)
    predicted_label_index = np.argmax(predicted_probabilities)
    predicted_label = enc.inverse_transform([[predicted_label_index]])[0][0]

    # Display the original and predicted emotion labels
    print(f"Original Label: {sample_label}")
    print(f"Predicted Label: {predicted_label}")

    # Display the audio for the selected sample
    Audio(sample_path)