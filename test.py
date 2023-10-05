from io import BytesIO
import streamlink
import subprocess
from pydub import AudioSegment
import numpy as np
import librosa
import tensorflow as tf

url = "https://www.twitch.tv/videos/1688065015"

streams = streamlink.streams(url)
audio_url = streams["audio"].url
cmd = f"ffmpeg -ss 01:20:00 -i {audio_url} -vn -acodec mp3 -t 3600 -f mp3 -"
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
audio_data, _ = process.communicate()


def split_audio(audio_data):
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

# def feature_extraction(audio_segments):
#     features = []
#
#     for segment in audio_segments:
#
#         sample, sample_rate = librosa.load(segment)
#         # sample = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
#         # sample_rate = segment.frame_rate
#         zero_cross_feat = np.mean(librosa.feature.zero_crossing_rate(y=sample))
#         mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=40)
#         mfccsscaled = np.mean(mfccs, axis=1)
#         mfccsscaled = np.append(mfccsscaled, zero_cross_feat)
#         mfccsscaled = mfccsscaled.reshape(1, -1)
#
#         features.append(mfccsscaled)
#
#     return features

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


audio_samples = split_audio(audio_data)
print(audio_samples)
sample_features = feature_extraction(audio_samples)
print(sample_features)

predictions = []

model = tf.keras.models.load_model("model_keras.h5")

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

for i, pred in enumerate(predictions):
    if pred["is_funny"]:
        print(pred)
