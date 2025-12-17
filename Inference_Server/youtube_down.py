from yt_dlp import YoutubeDL

FFMPEG_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
    'outtmpl': 'downloads/%(title)s.%(ext)s',
    'ffmpeg_location': FFMPEG_PATH  # ⚡ 여기 추가
}

url = "https://youtube.com/shorts/r7TtVHmRq54?si=34XD6wX7M2udD_Kg"

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
