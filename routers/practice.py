from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

import librosa
import numpy as np
import subprocess
import tempfile
import os
import io
import platform
import random
from pydub import AudioSegment

from database import get_db
from models import SavedWord

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

if platform.system() == "Windows":
    _FFMPEG_LOCAL = os.path.join(_PROJECT_ROOT, "voice_practice", "ffmpeg.exe")
    _FFMPEG = _FFMPEG_LOCAL if os.path.exists(_FFMPEG_LOCAL) else "ffmpeg"
else:
    _FFMPEG = "ffmpeg"

router = APIRouter(prefix="/api/practice")


class Task(BaseModel):
    word: str
    hakka: str = ""
    image_path: str
    audio_url: str


class ScoreResult(BaseModel):
    score: int
    message: str


def dtw_score_ssl(y_user: np.ndarray, y_ref: np.ndarray, sr: int) -> int:
    y_user, _ = librosa.effects.trim(y_user, top_db=15)
    y_user = y_user / (np.max(np.abs(y_user)) + 1e-6) * np.max(np.abs(y_ref))

    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
    mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr, n_mfcc=13)

    feat_ref = np.concatenate(
        [mfcc_ref, librosa.feature.delta(mfcc_ref), librosa.feature.delta(mfcc_ref, order=2)],
        axis=0
    )
    feat_user = np.concatenate(
        [mfcc_user, librosa.feature.delta(mfcc_user), librosa.feature.delta(mfcc_user, order=2)],
        axis=0
    )

    feat_ref = (feat_ref - np.mean(feat_ref)) / (np.std(feat_ref) + 1e-6)
    feat_user = (feat_user - np.mean(feat_user)) / (np.std(feat_user) + 1e-6)

    D, wp = librosa.sequence.dtw(X=feat_ref, Y=feat_user, metric="euclidean")
    avg_dist = D[-1, -1] / len(wp)

    user_len = feat_user.shape[1]
    ref_len = feat_ref.shape[1]
    coverage_ratio = user_len / ref_len
    path_deviation = len(wp) / max(user_len, ref_len)

    if avg_dist < 3.2:
        base_score = 100 - (avg_dist * 1.5)
    elif avg_dist < 6.0:
        base_score = 85 - (avg_dist - 3.2) * 15.0
    else:
        base_score = 10

    final_score = base_score

    user_vol = np.mean(np.std(feat_user, axis=1))
    ref_vol = np.mean(np.std(feat_ref, axis=1))
    vol_ratio = user_vol / (ref_vol + 1e-6)

    if vol_ratio < 0.85:
        final_score *= 0.4

    if path_deviation > 1.05:
        final_score *= 0.3

    if coverage_ratio < 0.80 or coverage_ratio > 1.30:
        penalty = min(coverage_ratio, 1 / coverage_ratio) ** 2
        final_score *= penalty

    print(
        f"\n[DEBUG] 距離: {avg_dist:.2f} | "
        f"覆蓋率: {coverage_ratio:.2f} | "
        f"偏差: {path_deviation:.2f} | "
        f"分數: {final_score}"
    )

    return int(np.clip(final_score, 0, 100))


@router.get("/task", response_model=Task)
async def get_task(
    word: str | None = None,
    user_id: int = 1,
    db: AsyncSession = Depends(get_db)
):
    query = select(SavedWord).where(SavedWord.user_id == user_id)

    if word:
        query = query.where(SavedWord.label_zh == word)

    result = await db.execute(query)
    rows = result.scalars().all()

    rows = [
        row for row in rows
        if row.audio_path and row.image_path
    ]

    if not rows:
        raise HTTPException(status_code=404, detail="找不到可練習的詞彙")

    row = random.choice(rows)

    return Task(
        word=row.label_zh,
        hakka=row.label_hakka,
        image_path=row.image_path,
        audio_url=row.audio_path
    )


@router.post("/score", response_model=ScoreResult)
async def score_recording(
    audio: UploadFile = File(...),
    word: str = Form(""),
    user_id: int = Form(1),
    db: AsyncSession = Depends(get_db)
):
    query = select(SavedWord).where(
        SavedWord.user_id == user_id,
        SavedWord.audio_path != ""
    )

    if word:
        query = query.where(SavedWord.label_zh == word)

    result = await db.execute(query)
    rows = result.scalars().all()

    if not rows:
        raise HTTPException(status_code=422, detail="找不到該詞彙的標準音")

    row = rows[0]
    ref_path = row.audio_path.lstrip("/")

    if not os.path.exists(ref_path):
        raise HTTPException(status_code=500, detail="標準音檔不存在")

    audio_bytes = await audio.read()

    try:
        audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        if audio_seg.dBFS < -40:
            return ScoreResult(score=0, message="偵測不到聲音，請靠近麥克風大聲練習喔！")
    except Exception:
        pass

    tmp_in_path = None
    tmp_out_path = None

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
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass

    try:
        y_ref, _ = librosa.load(ref_path, sr=16000)

        y_user, _ = librosa.effects.trim(y_user, top_db=30)
        y_ref, _ = librosa.effects.trim(y_ref, top_db=30)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"資源處理失敗：{e}")

    if len(y_user) == 0:
        return ScoreResult(score=0, message="偵測不到有效發音。")

    final_score = dtw_score_ssl(y_user, y_ref, sr)

    if final_score > 80:
        message = "太棒了！你的發音與標準音契合度極高。"
    elif final_score > 50:
        message = "表現不錯，請嘗試注意發音細節後再挑戰！"
    else:
        message = "差距明顯，建議先多聽幾次標準發音喔。"

    return ScoreResult(score=final_score, message=message)
