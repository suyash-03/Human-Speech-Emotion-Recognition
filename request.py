import requests

url = 'http://localhost:5000/predict'  # Updated endpoint to match your Flask app route
audio_file_path = './audio/audio.mp3'

# Assuming you are sending an audio file, adjust the content type accordingly
files = {'file': open(audio_file_path, 'rb')}
r = requests.post(url, files=files)

# Check the response
if r.status_code == 200:
    print(f"Prediction: {r.json()['prediction']}")
else:
    print(f"Error: {r.json()['error']}")