from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import librosa
import numpy as np
import subprocess
import tempfile
import os
import io
import platform
from pydub import AudioSegment

# --- 1. 環境與模型初始化 ---
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

if platform.system() == "Windows":
    _FFMPEG_LOCAL = os.path.join(_PROJECT_ROOT, "voice_practice", "ffmpeg.exe")
    _FFMPEG = _FFMPEG_LOCAL if os.path.exists(_FFMPEG_LOCAL) else "ffmpeg"
else:
    _FFMPEG = "ffmpeg"

router = APIRouter(prefix="/api/practice")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_practice", "data")
DB_PATH  = os.path.join(DATA_DIR, "db.csv")

def load_db() -> pd.DataFrame:
    return pd.read_csv(DB_PATH)


# --- 3. 升級版評分機制 (SSL + DTW + 指數加權) ---
def dtw_score_ssl(y_user: np.ndarray, y_ref: np.ndarray, sr: int) -> int:
    # 1. 前處理
    y_user, _ = librosa.effects.trim(y_user, top_db=15)
    y_user = y_user / (np.max(np.abs(y_user)) + 1e-6) * np.max(np.abs(y_ref))
    
    # 2. 提取 39 維動態特徵
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
    mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr, n_mfcc=13)
    
    feat_ref = np.concatenate([mfcc_ref, librosa.feature.delta(mfcc_ref), librosa.feature.delta(mfcc_ref, order=2)], axis=0)
    feat_user = np.concatenate([mfcc_user, librosa.feature.delta(mfcc_user), librosa.feature.delta(mfcc_user, order=2)], axis=0)
    
    feat_ref = (feat_ref - np.mean(feat_ref)) / (np.std(feat_ref) + 1e-6)
    feat_user = (feat_user - np.mean(feat_user)) / (np.std(feat_user) + 1e-6)

    # 3. 執行 DTW
    D, wp = librosa.sequence.dtw(X=feat_ref, Y=feat_user, metric='euclidean')
    avg_dist = D[-1, -1] / len(wp)

    # 4. 結構數據
    user_len = feat_user.shape[1]
    ref_len = feat_ref.shape[1]
    coverage_ratio = user_len / ref_len
    # 重要：偏差值 1.0 代表完美對齊，中文椅子雖然距離近，但時間結構通常會歪掉
    path_deviation = len(wp) / max(user_len, ref_len)

    # --- 5. 針對截圖數據重新定義門檻 ---
    if avg_dist < 3.2: # 只有低於 3.2 才是完美匹配
        base_score = 100 - (avg_dist * 1.5)
    elif avg_dist < 6.0:
        # 讓 3.36 (哈囉) 落在這個陡峭的降分區
        base_score = 85 - (avg_dist - 3.2) * 15.0 # 3.36 會變約 82.6 分
    else:
        base_score = 10

    # --- 6. 複合式懲罰 (Discriminator) --- 
    final_score = base_score

    # [新增] 波動率比對：區分中文平聲與客語起伏
    user_vol = np.mean(np.std(feat_user, axis=1))
    ref_vol = np.mean(np.std(feat_ref, axis=1))
    vol_ratio = user_vol / (ref_vol + 1e-6)

    if vol_ratio < 0.85: # 提高門檻，攔截波動較小的詞彙
        final_score *= 0.4

    # 調整路徑偏差門檻：將原本的 1.10 調得更嚴格 (視情況調整) 
    if path_deviation > 1.05: # 原本是 1.10
        final_score *= 0.3 

    # 原有的長度懲罰邏輯保持在下方... 
    if coverage_ratio < 0.80 or coverage_ratio > 1.30:
        penalty = min(coverage_ratio, 1/coverage_ratio) ** 2
        final_score *= penalty 

    print(f"\n[DEBUG] 距離: {avg_dist:.2f} | 覆蓋率: {coverage_ratio:.2f} | 偏差: {path_deviation:.2f} | 分數: {final_score}")
    return int(np.clip(final_score, 0, 100))

class Task(BaseModel):
    word:       str
    image_path: str
    audio_url:  str

class ScoreResult(BaseModel):
    score:   int
    message: str

@router.get("/task", response_model=Task)
def get_task(word: str | None = None):
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

    audio_bytes = await audio.read()
    
    # --- 新增：dBFS 靜音偵測 (封鎖沒說話拿高分漏洞) ---
    try:
        audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        if audio_seg.dBFS < -40:
            return ScoreResult(score=0, message="偵測不到聲音，請靠近麥克風大聲練習喔！")
    except Exception:
        pass # 解析失敗則交由後續處理

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
            if p and os.path.exists(p):
                try: os.unlink(p)
                except: pass

    # 載入標準音與切除靜音
    try:
        VP_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_practice")
        if not os.path.isabs(ref_path):
            ref_path = os.path.normpath(os.path.join(VP_DIR, ref_path))
        y_ref, _ = librosa.load(ref_path, sr=16000)
        
        y_user, _ = librosa.effects.trim(y_user, top_db=30)
        y_ref,  _ = librosa.effects.trim(y_ref,  top_db=30)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"資源處理失敗：{e}")

    # 二次檢查內容長度
    if len(y_user) == 0:
        return ScoreResult(score=0, message="偵測不到有效發音。")

    # 執行 SSL + DTW 評分
    final_score = dtw_score_ssl(y_user, y_ref, sr)

    if final_score > 80:
        message = "太棒了！你的發音與標準音契合度極高。"
    elif final_score > 50:
        message = "表現不錯，請嘗試注意發音細節後再挑戰！"
    else:
        message = "差距明顯，建議先多聽幾次標準發音喔。"

    return ScoreResult(score=final_score, message=message)
