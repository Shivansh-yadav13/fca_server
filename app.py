import json
import flask
import librosa
import numpy as np
from pytube import YouTube
import tensorflow as tf
from pydub import AudioSegment
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import streamlink
import subprocess
import base64
from supabase import create_client, Client
import threading

url: str = "https://mzwpeqplxjiupysnwteo.supabase.co"
key:str = os.getenv('supabase-key')

supabase: Client = create_client(url, key)

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

def fetch_audio_from_url(platform, url, start_timestamps):
    try:
        if platform == 'twitch':
            streams = streamlink.streams(url)
            if "audio" in streams:
                audio_url = streams["audio"].url
                cmd = f"ffmpeg -ss {start_timestamps['hour']:02d}:{start_timestamps['min']:02d}:{start_timestamps['sec']:02d} -i {audio_url} -vn -acodec mp3 -t 3600 -f mp3 -"
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                audio_data, _ = process.communicate()
                return audio_data
            else:
                return None
        elif platform == 'kick':
            cmd = f"ffmpeg -ss {start_timestamps['hour']:02d}:{start_timestamps['min']:02d}:{start_timestamps['sec']:02d} -i {url} -vn -acodec mp3 -t 3600 -f mp3 -"
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            audio_data, _ = process.communicate()
            return audio_data
        elif platform == 'youtube':
            yt = YouTube(url)
            video = yt.streams.get_highest_resolution()
            cmd = f'ffmpeg -ss {start_timestamps["hour"]:02d}:{start_timestamps["min"]:02d}:{start_timestamps["sec"]:02d} -i "{video.url}" -vn -acodec mp3 -t 3600 -f mp3 pipe:1'
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            audio_data, error_data = process.communicate()
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

def split_audio_v2(audio_data):
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


def background_processing(platform, url, start_timestamps, start_time_secs, user_id):
    audio = fetch_audio_from_url(platform, url, start_timestamps)
    if audio:
        audio_samples = split_audio_v2(audio)
        sample_features = feature_extraction(audio_samples)
        predictions = []
        print("background processing in progress")
        for i, x in enumerate(sample_features):
            pred = model.predict(x)
            is_funny = pred[0][0] > pred[0][1]
            if (bool(is_funny)):
                predictions.append({
                    "time_stamp": (i * 60) + start_time_secs,
                    "is_funny": bool(is_funny),
                    "funniness_score": float(pred[0][0]),
                    "boringness_score": float(pred[0][1]),
                })
        # Get the last request data
        last_request_data = supabase.table('users').select('last_request_data').eq('id', user_id).execute()
        # Set Server busy status to False
        supabase.table('users').update({"server_busy_status": False}).eq("id", user_id).execute()

        print(last_request_data)

        # Update last request data
        last_request_data = last_request_data.data[0]['last_request_data']
        last_request_data['last_clips'] = predictions
        supabase.table('users').update({"last_request_data": last_request_data}).eq("id", user_id).execute()

        # Reduce trial requests
        user_trial_requests = supabase.table('users').select('trial_requests').eq('id', user_id).execute()
        supabase.table('users').update({"trial_requests": (user_trial_requests - 1)}).eq('id', user_id).execute()



@app.route('/analyze_twitch_audio', methods=['POST'])
def analyze_twitch_audio():
    try:
        twitch_url = request.form['twitch_url']
        start_timestamps = request.form['start_timestamps']
        user_id = request.form['user_id']
        start_timestamps = json.loads(start_timestamps)
        start_time_secs = (start_timestamps['hour'] * 60 * 60) + (start_timestamps['min'] * 60) + start_timestamps['sec']
        supabase.table('users').update({"server_busy_status": True}).eq("id", user_id).execute()
        print("Starting the background process")
        background_process = threading.Thread(target=background_processing, args=("twitch", twitch_url, start_timestamps, start_time_secs, user_id))
        background_process.start()
        return jsonify({"message": "Background Task started"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/analyze_kick_audio', methods=['POST'])
def analyze_kick_audio():
    try:
        kick_m3u8_url = request.form['kick_m3u8_url']
        start_timestamps = request.form['start_timestamps']
        user_id = request.form['user_id']
        start_timestamps = json.loads(start_timestamps)
        start_time_secs = (start_timestamps['hour'] * 60 * 60) + (start_timestamps['min'] * 60) + start_timestamps['sec']
        supabase.table('users').update({"server_busy_status": True}).eq("id", user_id).execute()
        background_process = threading.Thread(target=background_processing, args=("kick", kick_m3u8_url, start_timestamps, start_time_secs, user_id))
        background_process.start()
        return jsonify({"message": "Background Task started"})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analyze_youtube_audio', methods=['POST'])
def analyze_youtube_audio():
    try:
        youtube_url = request.form['youtube_url']
        start_timestamps = request.form['start_timestamps']
        user_id = request.form['user_id']
        start_timestamps = json.loads(start_timestamps)
        start_time_secs = (start_timestamps['hour'] * 60 * 60) + (start_timestamps['min'] * 60) + start_timestamps['sec']
        supabase.table('users').update({"server_busy_status": True}).eq("id", user_id).execute()
        background_process = threading.Thread(target=background_processing,args=("youtube", youtube_url, start_timestamps, start_time_secs, user_id))
        background_process.start()
        return jsonify({"message": "Background Task started"})
    except Exception as e:
        return jsonify({'error': str(e)})

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
    url = request.form['url']
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
        # cmd = f"ffmpeg -ss {hours:02d}:{minutes:02d}:{secs:02d} -i {video_url} -t 60 -c:v copy -c:a aac -async 1 -strict experimental -f avi -"
        # cmd = f"ffmpeg -ss {hours:02d}:{minutes:02d}:{secs:02d} -i {video_url} -t 60 -c:v libvpx -c:a libvorbis -async 1 -f webm -"
        # cmd = f"ffmpeg -ss {hours:02d}:{minutes:02d}:{secs:02d} -i {video_url} -t 60 -c:v copy -c:a aac -async 1 -f matroska -"
        # cmd = f"ffmpeg -ss {hours:02d}:{minutes:02d}:{secs:02d} -i {video_url} -t 60 -c:v wmv2 -c:a wmav2 -async 1 -f asf -"
        cmd = f"ffmpeg -ss {hours:02d}:{minutes:02d}:{secs:02d} -i {video_url} -t 60 -c:v libx264 -c:a aac -async 1 -f flv -"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        video_data, error = process.communicate()

        video_data_base64 = base64.b64encode(video_data).decode('utf-8')
        response = flask.Response(response=video_data_base64, content_type='application/octet-stream')
        return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
