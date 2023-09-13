import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import streamlink
import tempfile
import os

app = Flask(__name__)
cors = CORS(app, resources={r"/analyze_video": {"origins": ["http://localhost:3000", "https://fusionclips.pro"]}})

# Load your pre-trained model here
model = tf.keras.models.load_model("model_keras.h5")

def fetch_audio_from_twitch(url):
    fd, temp_audio_file_path = tempfile.mkstemp(suffix=".mp3")
    file_name = os.path.basename(temp_audio_file_path)
    os.close(fd)
    try:
        streams = streamlink.streams(url)
        if "audio" in streams:
            print("Inside IF Audio")
            audio_url = streams["audio"].url
            print("Audio URL:", audio_url)
            os.system(f"ffmpeg -i {audio_url} -vn -acodec mp3 -t 3600 {file_name}")
            # os.system(f"ffmpeg -ss 01:20:00 -i {audio_url} -vn -acodec mp3 -t 600 {temp_audio_file_name}")
            audio_segment = AudioSegment.from_file(file_name, format="mp3")
            try:
                temp_path = tempfile.gettempdir()
                os.remove(os.path.join(temp_path, file_name))
                print(f"File '{file_name}' has been removed.")
            except FileNotFoundError:
                print(f"File '{file_name}' does not exist.")
            except Exception as e:
                print(f"An error occurred while trying to remove '{file_name}': {str(e)}")
            return audio_segment
        else:
            return None
    except Exception as e:
        return None


def extract_audio_from_video(video_file):
    try:
        audio = AudioSegment.from_file(video_file, format="mp4")  # You can adjust the format as needed
        return audio
    except Exception as e:
        raise e

def split_audio(audio):
    segment_duration = 60 * 1000
    segments = [audio[i:i + segment_duration] for i in range(0, len(audio), segment_duration)]
    segment_bytesio_list = []

    for i, segment in enumerate(segments):
        # Export the audio segment to BytesIO
        segment_bytesio = BytesIO()
        segment.export(segment_bytesio, format="wav")  # You can choose other supported formats if needed
        segment_bytesio.seek(0)  # Reset the BytesIO position to the beginning
        segment_bytesio_list.append(segment_bytesio)

    return segment_bytesio_list


def feature_extraction(audio_samples):
    features = []

    for audio_sample in audio_samples:
        audio = librosa.load(audio_sample)
        sample = audio[0]
        sample_rate = audio[1]
        zero_cross_feat = librosa.feature.zero_crossing_rate(sample).mean()
        mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        mfccsscaled = np.append(mfccsscaled, zero_cross_feat)
        mfccsscaled = mfccsscaled.reshape(1, 41, )
        features.append(mfccsscaled)

    return features

@app.route('/analyze_twitch_audio', methods=['POST'])
def analyze_twitch_audio():
    try:
        twitch_url = request.form['twitch_url']
        if twitch_url:
            audio = fetch_audio_from_twitch(twitch_url)
            print(audio)
            if audio:
                audio_samples = split_audio(audio)
                sample_features = feature_extraction(audio_samples)
                predictions = []
                for i, x in enumerate(sample_features):
                    pred = model.predict(x)
                    is_funny = pred[0][0] > pred[0][1]
                    predictions.append({
                        "section": f"{i}:00 - {i + 1}:00",
                        "is_funny": bool(is_funny),
                        "funniness_score": float(pred[0][0]),
                        "boringness_score": float(pred[0][1]),
                    })
                return jsonify(predictions)
            else:
                return jsonify({"error": "Failed to fetch audio from the Twitch URL"})
        else:
            return jsonify({"error": "No Twitch URL provided"})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    try:
        video_file = request.files['video']
        if video_file:
            # Extract audio from the uploaded video file
            video_bytes = video_file.read()
            audio = extract_audio_from_video(BytesIO(video_bytes))

            # Split the audio into segments and process them
            audio_samples = split_audio(audio)
            sample_features = feature_extraction(audio_samples)

            predictions = []

            for i, x in enumerate(sample_features):
                # Assuming you have a trained model for audio analysis
                pred = model.predict(x)
                is_funny = pred[0][0] > pred[0][1]
                predictions.append({
                    "section": f"{i}:00 - {i + 1}:00",
                    "is_funny": bool(is_funny),
                    "funniness_score": float(pred[0][0]),
                    "boringness_score": float(pred[0][1]),
                })
            print(predictions)
            return jsonify(predictions)
        else:
            return jsonify({"error": "No video file provided"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    try:
        audio_file = request.files['audio']
        if audio_file:
            # Convert the uploaded audio to an AudioSegment
            audio = AudioSegment.from_file(audio_file, format="mp3")
            audio_samples = split_audio(audio)
            sample_features = feature_extraction(audio_samples)

            predictions = []

            for i, x in enumerate(sample_features):
                pred = model.predict(x)
                is_funny = pred[0][0] > pred[0][1]
                predictions.append({
                    "section": f"{i}:00 - {i + 1}:00",
                    "is_funny": bool(is_funny),
                    "funniness_score": float(pred[0][0]),
                    "boringness_score": float(pred[0][1]),
                })

            return jsonify(predictions)
        else:
            return jsonify({"error": "No audio file provided"})
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
