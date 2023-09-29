import json

import flask
import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import streamlink
import subprocess
import base64

app = Flask(__name__)
cors = CORS(app,
            resources={
                r"/analyze_video": {
                    "origins": ["http://localhost:3000", "https://fusionclips.pro"]
                },
                r"/analyze_twitch_audio": {
                    "origins": ["http://localhost:3000", "https://fusionclips.pro"]
                },
                r"/download_clip": {
                    "origins": ['http://localhost:3000', 'https://fusionclips.pro']
                }
            })

# Load your pre-trained model here
model = tf.keras.models.load_model("model_keras.h5")

def fetch_audio_from_twitch(url, start_timestamps):
    try:
        streams = streamlink.streams(url)
        if "audio" in streams:
            audio_url = streams["audio"].url
            cmd = f"ffmpeg -ss {start_timestamps['hour']}:{start_timestamps['min']}:{start_timestamps['sec']} -i {audio_url} -vn -acodec mp3 -t 3600 -f mp3 -"
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            audio_data, _ = process.communicate()
            # os.system(f"ffmpeg -ss 01:20:00 -i {audio_url} -vn -acodec mp3 -t 600 {temp_audio_file_name}")
            return audio_data
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

def split_twitch_audio(audio_data):
    segment_duration = 60 * 1000  # 60 seconds in milliseconds
    audio = AudioSegment.from_mp3(BytesIO(audio_data))

    # Set sample width to 2 bytes for 16-bit PCM audio (floating-point format)
    audio = audio.set_sample_width(2)

    # Split the audio into segments
    segments = []
    for start_time in range(0, len(audio), segment_duration):
        end_time = start_time + segment_duration
        segment = audio[start_time:end_time]

        segment_bytesio = BytesIO()
        segment.export(segment_bytesio, format="wav")
        segment_bytesio.seek(0)

        segments.append(segment_bytesio)

    return segments

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
        sample, sample_rate = librosa.load(audio_sample)
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
        start_timestamps = request.form['start_timestamps']
        start_timestamps = json.loads(start_timestamps)
        start_time_secs = (start_timestamps['hour'] * 60 * 60) + (start_timestamps['min'] * 60) + start_timestamps['sec']
        if twitch_url:
            audio = fetch_audio_from_twitch(twitch_url, start_timestamps)
            if audio:
                audio_samples = split_twitch_audio(audio)
                sample_features = feature_extraction(audio_samples)
                predictions = []
                for i, x in enumerate(sample_features):
                    pred = model.predict(x)
                    is_funny = pred[0][0] > pred[0][1]
                    predictions.append({
                        "time_stamp": (i*60) + start_time_secs,
                        "is_funny": bool(is_funny),
                        "funniness_score": float(pred[0][0]),
                        "boringness_score": float(pred[0][1]),
                    })
                    # add another model (gaming/shooting detection wala)
                    # if the is_funny -> false, then pass it through the shooting model
                    # and then if that has engagement then pass clip
                    # also a tag will be passed to enable or disable shooting model prediction
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

@app.route('/download_clip', methods=['POST'])
def download_clip():
    url = request.form['twitch_url']
    start_seconds = request.form['start_timestamps_sec']
    start_seconds = int(start_seconds)
    end_seconds = start_seconds + 60
    hours = start_seconds // 3600
    minutes = (start_seconds % 3600) // 60
    secs = start_seconds % 60
    hours_end = end_seconds // 3600
    minutes_end = (end_seconds % 3600) // 60
    secs_end = end_seconds % 60
    streams = streamlink.streams(url)
    if "720p60" in streams:
        video_url = streams["720p60"].url
        cmd = f"ffmpeg -ss {hours:02d}:{minutes:02d}:{secs:02d} -to {hours_end:02d}:{minutes_end:02d}:{secs_end:02d} -i {video_url} -c:v copy -c:a aac -strict experimental -f avi -"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        video_data, error = process.communicate()

        video_data_base64 = base64.b64encode(video_data).decode('utf-8')
        response = flask.Response(response=video_data_base64, content_type='application/octet-stream')
        return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
