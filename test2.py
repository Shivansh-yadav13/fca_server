import subprocess
import streamlink
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.plugin.api import validate
from streamlink.stream import HLSStream, HTTPStream
from streamlink.utils.parse import parse_json
from streamlink.exceptions import PluginError
from streamlink import Streamlink


new_Str = Streamlink()

new_Str.load_plugins("kick.py")

some = HLSStream.parse_variant_playlist(new_Str, "https://stream.kick.com/ivs/v1/196233775518/FSSERpXRQg7Y/2023/10/2/4/57/NHiVXj0CIUwA/media/hls/720p60/playlist.m3u8")
print(some)

# url = "https://www.twitch.tv/videos/1922395449"
# url = "https://www.youtube.com/watch?v=JuYG4-byXJk"
    # start_timestamps = request.form['start_timestamps']
    # start_timestamps = json.loads(start_timestamps)
# streams = streamlink.streams(url)
# streamlink.
# print(streams)
# if "720p60" in streams:
#     video_url = streams["720p60"].url
# print(video_url)
# cmd = f"ffmpeg -i https://stream.kick.com/ivs/v1/196233775518/FSSERpXRQg7Y/2023/10/2/4/57/NHiVXj0CIUwA/media/hls/720p60/playlist.m3u8 -vn -acodec mp3 -t 60 -f mp3 -"
# process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
# audio_data, error = process.communicate()
# print(error)
# print(audio_data)
