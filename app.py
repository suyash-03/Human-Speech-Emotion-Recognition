import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import librosa


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


# Load the pre-trained model
# pretrained_model = create_LSTM()
# pretrained_model.load_weights('./saved_weights/speech_classification.h5')
# pretrained_model.trainable = False

# Load the saved model
loaded_model = load_model('./saved_weights/speech_classification.h5')

# Now, you can use the loaded_model for making predictions on new data
# For example, if you have new data in X_test, you can do:


# Specify the directory to save uploaded audio files
UPLOAD_FOLDER = 'audio'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_features = mfcc_features.T
    mfcc_features = np.mean(mfcc_features, axis=0)
    reshaped_features = np.expand_dims(mfcc_features, axis=-1)
    return reshaped_features


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If the user does not select a file, the browser may send an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        current_dir = os.getcwd()
        current_dir = current_dir + r'\audio'
        file_path = os.path.join('audio', 'audio.mp3')
        file.save(file_path)

        # Preprocess the audio file
        processed_audio = preprocess_audio(file_path)
        print(processed_audio.shape)

        # Make prediction using the pre-trained model
        input_data = np.array([processed_audio])
        prediction = loaded_model.predict(input_data)

        # Process the prediction as needed
        print(prediction)
        max_index = np.argmax(prediction[0])

        emotion_map = {'fear': 0, 'angry': 1, 'disgust': 2, 'neutral': 3, 'sad': 4, 'ps': 5, 'happy': 6}
        reversed_dict = {value: key for key, value in emotion_map.items()}

        return jsonify({'prediction': reversed_dict[max_index]})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
