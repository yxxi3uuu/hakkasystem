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

# ── 內建預設詞彙 ──
# audio_file 指向本地快取路徑；若不存在會在 lifespan 時自動用 TTS API 產生
_VP_BASE = "voice_practice/data"
PRESET_WORDS = [
    {
        "word": "蘋果",
        "hakka": "頻果",
        "image_path": "/voice_practice/images/images.jpg",
        "audio_file": os.path.join(_VP_BASE, "audios", "apple.wav"),
        "audio_url":  "/voice_practice/audios/apple.wav",
    },
    {
        "word": "椅子",
        "hakka": "椅仔",
        "image_path": "/voice_practice/images/800x.jpg",
        "audio_file": os.path.join(_VP_BASE, "audios", "chair.wav"),
        "audio_url":  "/voice_practice/audios/chair.wav",
    },
    {
        "word": "電視",
        "hakka": "電視",
        "image_path": "/voice_practice/images/Samsung_LE26R41BD_and_Yamada_DVD_player_20030624.jpg",
        "audio_file": os.path.join(_VP_BASE, "audios", "TV.wav"),
        "audio_url":  "/voice_practice/audios/TV.wav",
    },
]
PRESET_BY_WORD = {p["word"]: p for p in PRESET_WORDS}


async def ensure_preset_audios():
    """
    啟動時檢查預設詞彙的音檔是否存在。
    若不存在，呼叫客語 TTS API 產生並存到 voice_practice/data/audios/。
    """
    from routers.hakka_api import get_tts_token, generate_hakka_tts
    import shutil

    os.makedirs(os.path.join(_VP_BASE, "audios"), exist_ok=True)

    for preset in PRESET_WORDS:
        target = preset["audio_file"]
        if os.path.exists(target) and os.path.getsize(target) > 1000:
            print(f"[Practice] 預設音檔已存在：{target}")
            continue

        print(f"[Practice] 產生預設音檔：{preset['word']} ({preset['hakka']}) → {target}")
        try:
            # generate_hakka_tts 會存到 static/audios/words/ 並回傳路徑
            # 我們把它複製到 voice_practice/data/audios/
            tmp_path = await generate_hakka_tts(preset["hakka"], folder="words")
            src = tmp_path.lstrip("/")   # 去掉開頭的 /
            if os.path.exists(src):
                shutil.copy2(src, target)
                print(f"[Practice] ✅ 已產生：{target}")
            else:
                print(f"[Practice] ❌ TTS 回傳路徑不存在：{src}")
        except Exception as e:
            print(f"[Practice] ❌ 產生音檔失敗 ({preset['word']}): {e}")


class Task(BaseModel):
    word: str
    hakka: str = ""
    pinyin: str = ""
    image_path: str
    audio_url: str


class ScoreResult(BaseModel):
    score: int
    message: str


def dtw_score_ssl(y_user: np.ndarray, y_ref: np.ndarray, sr: int) -> tuple[int, str]:
    # 1. 前處理 (保持原本寫好的高鑑別度設定)
    y_user, _ = librosa.effects.trim(y_user, top_db=15)
    y_user = y_user / (np.max(np.abs(y_user)) + 1e-6) * np.max(np.abs(y_ref)) 

    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13) 
    mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr, n_mfcc=13) 

    feat_ref = np.concatenate([mfcc_ref, librosa.feature.delta(mfcc_ref), librosa.feature.delta(mfcc_ref, order=2)], axis=0) 
    feat_user = np.concatenate([mfcc_user, librosa.feature.delta(mfcc_user), librosa.feature.delta(mfcc_user, order=2)], axis=0) 

    feat_ref = (feat_ref - np.mean(feat_ref)) / (np.std(feat_ref) + 1e-6) 
    feat_user = (feat_user - np.mean(feat_user)) / (np.std(feat_user) + 1e-6) 

    D, wp = librosa.sequence.dtw(X=feat_ref, Y=feat_user, metric="euclidean") 
    avg_dist = D[-1, -1] / len(wp)

    user_len = feat_user.shape[1] 
    ref_len = feat_ref.shape[1] 
    coverage_ratio = user_len / ref_len 
    path_deviation = len(wp) / max(user_len, ref_len) 

    # 分數映射邏輯 
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

    # --- 🌟 核心升級：物理能量音節切分與分析 🌟 ---
    # 利用短時能量 (RMS) 分析標準音有幾個音節起伏
    rms_ref = librosa.feature.rms(y=y_ref)[0]
    threshold = np.max(rms_ref) * 0.3
    voiced_frames = np.where(rms_ref > threshold)[0]
    
    peaks = []
    if len(voiced_frames) > 0:
        current_peak = [voiced_frames[0]]
        for f in voiced_frames[1:]:
            if f - current_peak[-1] > 5:  # 空隙大於 5 幀視為不同音節
                peaks.append(int(np.mean(current_peak)))
                current_peak = [f]
            else:
                current_peak.append(f)
        peaks.append(int(np.mean(current_peak)))

    num_syllables = len(peaks)
    bad_segments = []

    # 如果有明顯多個音節，檢查使用者在哪個區間表現較差
    if num_syllables > 1:
        for i, peak_frame in enumerate(peaks):
            seg_dists = [
                np.linalg.norm(feat_ref[:, r] - feat_user[:, u]) 
                for r, u in wp if abs(r - peak_frame) < (ref_len / (num_syllables * 2))
            ]
            if seg_dists and np.mean(seg_dists) > 4.5:
                bad_segments.append(i + 1)

    # 組合死板的診斷數據報告，留給 AI 當作 Prompt 輸入
    if final_score < 45:
        diagnostic_report = "整體語速、音調或發音嚴重偏離標準音節節奏。"
    elif len(bad_segments) == 0:
        diagnostic_report = "所有音節的發音、咬字與音調起伏皆完美契合標準音。"
    else:
        seg_str = "、".join([f"第 {x} 個音節" for x in bad_segments])
        diagnostic_report = f"大部分發音良好，但偵測到【{seg_str}】的音調或咬字有些許瑕疵。"

    print(f"[DEBUG] 總分: {final_score:.1f} | 物理診斷報告: {diagnostic_report}")
    
    # 傳回最終分數與診斷數據報告
    return int(np.clip(final_score, 0, 100)), diagnostic_report


@router.get("/presets")
async def get_preset_words():
    """回傳內建預設詞彙列表"""
    return [
        {"word": p["word"], "hakka": p["hakka"], "audio_url": p["audio_url"], "image_path": p["image_path"]}
        for p in PRESET_WORDS
    ]


@router.get("/task", response_model=Task)
async def get_task(
    word: str | None = None,
    user_id: int = 1,
    db: AsyncSession = Depends(get_db)
):
    # 1. 先嘗試從使用者的 saved_words 找
    query = select(SavedWord).where(SavedWord.user_id == user_id)
    if word:
        query = query.where(SavedWord.label_zh == word)

    result = await db.execute(query)
    rows = result.scalars().all()
    rows = [row for row in rows if row.audio_path and row.image_path]

    if rows:
        row = random.choice(rows)
        return Task(
            word=row.label_zh,
            hakka=row.label_hakka,
            pinyin=getattr(row, "label_pinyin", ""),
            image_path=row.image_path,
            audio_url=row.audio_path
        )

    # 2. Fallback：使用內建預設詞彙
    if word and word in PRESET_BY_WORD:
        p = PRESET_BY_WORD[word]
        return Task(word=p["word"], hakka=p["hakka"], image_path=p["image_path"], audio_url=p["audio_url"])

    if word:
        # 指定了單字但找不到（既不在 saved_words 也不在預設）
        raise HTTPException(status_code=404, detail=f"找不到「{word}」的練習資料")

    # 隨機從預設詞彙選一個
    p = random.choice(PRESET_WORDS)
    return Task(word=p["word"], hakka=p["hakka"], image_path=p["image_path"], audio_url=p["audio_url"])

class ScoreResult(BaseModel):
    score: int 
    message: str 
    ai_advice: str  

# 初始化 Gemini 客戶端 (請記得在環境變數或程式中設定你的 API 金鑰)
try:
    from google import genai as _genai
    _gemini_key = os.getenv("GEMINI_API_KEY", "")
    if _gemini_key:
        ai_client = _genai.Client(api_key=_gemini_key)
    else:
        ai_client = None
        print("[Gemini] GEMINI_API_KEY 未設定，語音評分將使用 fallback")
except ImportError:
    _genai = None
    ai_client = None
    print("[Gemini] google-genai 未安裝，語音評分將使用 fallback")

@router.post("/score", response_model=ScoreResult)
async def score_recording(
    audio: UploadFile = File(...),
    word: str = Form(""),
    user_id: int = Form(1),
    db: AsyncSession = Depends(get_db)
):
    ref_path = None

    # 1. 先從使用者 saved_words 找標準音
    if word:
        query = select(SavedWord).where(
            SavedWord.user_id == user_id,
            SavedWord.audio_path != "",
            SavedWord.label_zh == word
        )
        result = await db.execute(query)
        rows = result.scalars().all()
        if rows:
            ref_path = rows[0].audio_path.lstrip("/")

    # 2. Fallback：用內建預設詞彙的音檔
    if not ref_path and word and word in PRESET_BY_WORD:
        ref_path = PRESET_BY_WORD[word]["audio_file"]

    # 3. 還是找不到：從所有 saved_words 隨機選一個
    if not ref_path:
        query = select(SavedWord).where(
            SavedWord.user_id == user_id,
            SavedWord.audio_path != ""
        )
        result = await db.execute(query)
        rows = result.scalars().all()
        if rows:
            ref_path = rows[0].audio_path.lstrip("/")

    if not ref_path:
        raise HTTPException(status_code=422, detail="找不到該詞彙的標準音")

    if not os.path.exists(ref_path):
        raise HTTPException(status_code=500, detail=f"標準音檔不存在：{ref_path}")

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
        return ScoreResult(score=0, message="偵測不到有效發音。", ai_advice="請靠近麥克風再試一次。")

    # 執行我們升級後的 DTW 演算法，拿到分數與數據報告
    final_score, diagnostic_report = dtw_score_ssl(y_user, y_ref, sr)

    if final_score > 80:
        message = "太棒了！你的發音非常標準。"
    elif final_score > 50:
        message = "表現不錯，請嘗試注意發音細節後再挑戰！"
    else:
        message = "差距明顯，建議先多聽幾次標準發音喔。"
        
    try:
        if not ai_client:
            raise ValueError("Gemini client not available")
        prompt = f"""
        你是一位極具親和力且溫柔的台灣客家話家教老師。
        有一位學生剛剛練習了這個客語詞彙：『{word}』。
        
        後端聲學演算法給出的客觀評分與數據報告如下：
        - 學生得分：{final_score} 分
        - 演算法診斷結果：{diagnostic_report}
        
        請根據以上數據，用 30 個字以內、親切多變的口吻，給予學生一句具體的發音優化建議或鼓勵。
        嚴格限制：不要重複演算法專有名詞，直接告訴他哪裡好、哪裡需要微調即可。
        """
        
        response = ai_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
        )
        ai_advice = response.text.strip()
    except Exception as e:
        # 保底機制：如果 API 網路有問題，就使用原本的診斷報告
        ai_advice = f"{diagnostic_report}"

    return ScoreResult(score=final_score, message=message, ai_advice=ai_advice)
