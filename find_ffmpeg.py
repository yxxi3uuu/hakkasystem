"""找出系統上可用的 ffmpeg 路徑"""
import os
import subprocess
import sys

def try_ffmpeg(path):
    try:
        r = subprocess.run([path, "-version"], capture_output=True, timeout=3)
        if r.returncode == 0:
            return True
    except Exception:
        pass
    return False

# 1. 系統 PATH
if try_ffmpeg("ffmpeg"):
    print("ffmpeg 在系統 PATH 中可用")
    sys.exit(0)

# 2. imageio-ffmpeg (pip 套件內建)
try:
    import imageio_ffmpeg
    path = imageio_ffmpeg.get_ffmpeg_exe()
    if try_ffmpeg(path):
        print(f"imageio-ffmpeg: {path}")
        sys.exit(0)
except ImportError:
    print("imageio-ffmpeg 未安裝")

# 3. 常見 Windows 安裝路徑
candidates = [
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
]
for c in candidates:
    if os.path.exists(c) and try_ffmpeg(c):
        print(f"找到: {c}")
        sys.exit(0)

print("找不到 ffmpeg，需要安裝")
