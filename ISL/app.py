import os
import atexit
import numpy as np
import cv2
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import speech_recognition as sr
import string
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.stem import PorterStemmer
# import nltk
# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')



# Set TensorFlow environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

app = Flask(__name__)

# Load the sign language model
model = load_model('sign_language_model.h5')
label_map = {0: 'Hello', 1: 'ILoveYou', 2: 'Yes'}
image_size = 300

# Initialize NLP tools
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()


# Initialize the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Error: Unable to access the camera.")

detector = HandDetector(maxHands=1)

# Release camera resources on exit
@atexit.register
def cleanup():
    camera.release()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            hands, frame = detector.findHands(frame)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgCrop = frame[y:y + h, x:x + w]

                if imgCrop.size > 0:
                    imgCrop = cv2.resize(imgCrop, (image_size, image_size))
                    imgCrop = imgCrop / 255.0
                    imgCrop = np.expand_dims(imgCrop, axis=0)

                    predictions = model.predict(imgCrop)
                    class_index = np.argmax(predictions)
                    sign = label_map[class_index]

                    cv2.putText(frame, sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Encode and yield the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to serve video files
@app.route('/video/<path:filename>')
def serve_video(filename):
    if os.path.exists(f'static/letters_avatar_blender/{filename}'):
        return send_from_directory('static/letters_avatar_blender', filename)
    elif os.path.exists(f'static/phrases/{filename}'):
        return send_from_directory('static/phrases', filename)
    else:
        return jsonify({"error": "Video not found."}), 404

# Process the input text to return video paths or a pause for spaces
def process_input(input_text):
    arr = list(string.ascii_lowercase)
    input_text = input_text.lower()

    video_paths = []
    phrase_video_address = f'static/phrases/{input_text}.mp4'
    if os.path.exists(phrase_video_address):
        video_paths.append(phrase_video_address)
        return video_paths

    for char in input_text:
        if char in arr:
            video_address = f'static/letters_avatar_blender/{char}.mp4'
            if os.path.exists(video_address):
                video_paths.append(video_address)
            else:
                return f"Video for '{char}' not found."  # Return an error message as a string
        elif char == ' ':
            video_paths.append('pause')
        else:
            return f"'{char}' is not a valid character."  # Return an error message as a string

    return video_paths


# Process the input text to return video paths or a pause for spaces
# def process_input(input_text):
#     # Tokenization
#     tokens = word_tokenize(input_text.lower())

#     # Stopword removal
#     filtered_tokens = [word for word in tokens if word not in stop_words]

#     # Lemmatization
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

#     # Stemming
#     stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]

#     arr = list(string.ascii_lowercase)
#     video_paths = []

#     for token in stemmed_tokens:
#         if len(token) == 1 and token in arr:  # Single letters
#             video_address = f'static/letters_avatar_blender/{token}.mp4'
#             if os.path.exists(video_address):
#                 video_paths.append(video_address)
#             else:
#                 return f"Video for '{token}' not found."  # Return an error message as a string
#         elif len(token) > 1:  # Whole words
#             phrase_video_address = f'static/phrases/{token}.mp4'
#             if os.path.exists(phrase_video_address):
#                 video_paths.append(phrase_video_address)
#             else:
#                 return f"Video for phrase '{token}' not found."  # Return an error message as a string
#         elif token == ' ':
#             video_paths.append('pause')
#         else:
#             return f"'{token}' is not a valid character."  # Return an error message as a string

#     return video_paths

# Route to serve the homepage
@app.route('/')
def index():
    return render_template('homepage.html')

# Route to serve the text to sign page
@app.route('/texttosign')
def text_to_sign():
    return render_template('texttosign.html')

# Route to serve the sign to text page
@app.route('/signToText')
def sign_to_text():
    return render_template('signToText.html')

# Route for video feed from the camera
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle text input
@app.route('/text_input', methods=['POST'])
def text_input():
    input_text = request.form['input_text']
    video_paths = process_input(input_text)

    # Check if video_paths is a list
    if isinstance(video_paths, list):
        return jsonify({"videos": video_paths})
    else:
        # If video_paths is not a list, it means an error occurred
        return jsonify({"error": video_paths}), 400  # Return the error message with a 400 status code
# Route to handle audio input
@app.route

# Route to handle audio input
@app.route('/audio_input', methods=['POST'])
def audio_input():
    r = sr.Recognizer()
    audio_file = request.files['audio']
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        try:
            recognized_text = r.recognize_google(audio_data)
            video_paths = process_input(recognized_text.lower())
            return jsonify({"videos": video_paths})
        except sr.UnknownValueError:
            return jsonify({"error": "Sorry, I could not understand the audio."})
        except sr.RequestError as e:
            return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"})

if __name__ == '__main__':
    app.run(debug=True)