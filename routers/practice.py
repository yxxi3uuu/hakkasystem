from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import subprocess
import tempfile
import os

# 優先用專案內的 ffmpeg.exe，找不到才用系統的
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_FFMPEG_LOCAL = os.path.join(_PROJECT_ROOT, "voice_practice", "ffmpeg.exe")
_FFMPEG = _FFMPEG_LOCAL if os.path.exists(_FFMPEG_LOCAL) else "ffmpeg"

router = APIRouter(prefix="/api/practice")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_practice", "data")
DB_PATH  = os.path.join(DATA_DIR, "db.csv")

# ── 載入題庫 ──
def load_db() -> pd.DataFrame:
    return pd.read_csv(DB_PATH)

# ── 特徵提取 ──
def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc,   axis=1),
        np.mean(delta,  axis=1),
        np.mean(delta2, axis=1),
    ])

class Task(BaseModel):
    word:       str
    image_path: str   # 前端可用此路徑顯示圖片
    audio_url:  str   # 前端播放標準音的 URL

class ScoreResult(BaseModel):
    score:   int
    message: str

@router.get("/task", response_model=Task)
def get_task(word: str | None = None):
    """隨機（或指定）取一道練習題"""
    try:
        df = load_db()
    except Exception:
        raise HTTPException(status_code=500, detail="找不到題庫檔案")

    if word:
        row = df[df["word"] == word]
        if row.empty:
            raise HTTPException(status_code=404, detail="找不到該詞彙")
        row = row.iloc[0]
    else:
        row = df.sample(n=1).iloc[0]

    # 把本地路徑轉成可供前端存取的 URL
    audio_filename = os.path.basename(row["audio_path"])
    image_filename = os.path.basename(row["image_path"])

    return Task(
        word=row["word"],
        image_path=f"/voice_practice/images/{image_filename}",
        audio_url=f"/voice_practice/audios/{audio_filename}",
    )

@router.post("/score", response_model=ScoreResult)
async def score_recording(
    audio: UploadFile = File(...),
    word:  str        = "",
):
    """接收使用者錄音，與標準音比對後回傳分數"""
    try:
        df = load_db()
    except Exception:
        raise HTTPException(status_code=500, detail="找不到題庫檔案")

    if word:
        row = df[df["word"] == word]
        if row.empty:
            raise HTTPException(status_code=422, detail="找不到該詞彙的標準音")
        ref_path = row.iloc[0]["audio_path"]
    else:
        ref_path = df.sample(n=1).iloc[0]["audio_path"]

    # 解析上傳的音訊（直接呼叫 ffmpeg 轉成 wav）
    audio_bytes = await audio.read()
    tmp_in_path = tmp_out_path = None
    try:
        suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path + ".wav"
        result = subprocess.run(
            [_FFMPEG, "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_out_path],
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="ignore"))
        y_user, sr = librosa.load(tmp_out_path, sr=16000)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"音訊解析失敗：{e}")
    finally:
        for p in [tmp_in_path, tmp_out_path]:
            if p:
                try: os.unlink(p)
                except: pass

    # 載入標準音
    try:
        # CSV 裡的路徑是 "data/audios/xxx.wav"，相對於 voice_practice 目錄
        VP_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_practice")
        if not os.path.isabs(ref_path):
            ref_path = os.path.normpath(os.path.join(VP_DIR, ref_path))
        y_ref, _ = librosa.load(ref_path, sr=16000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"標準音載入失敗：{e}")

    # 切除靜音
    y_user, _ = librosa.effects.trim(y_user, top_db=20)
    y_ref,  _ = librosa.effects.trim(y_ref,  top_db=20)

    # 特徵比對
    feat_ref  = extract_features(y_ref,  sr)
    feat_user = extract_features(y_user, sr)
    raw_sim   = max(0.0, 1 - cosine(feat_ref, feat_user))

    # 非線性縮放：把 0~1 的相似度映射到較合理的分數
    # 用 ^2 取代 ^5，讓分數分布更合理
    final_score = int(min(max(raw_sim ** 2 * 100, 0), 100))

    if final_score > 85:
        message = "太棒了！你的發音與標準音契合度極高。"
    elif final_score > 60:
        message = "表現不錯，請嘗試注意發音細節後再挑戰！"
    else:
        message = "差距較大，建議先多聽幾次標準發音喔。"

    return ScoreResult(score=final_score, message=message)
