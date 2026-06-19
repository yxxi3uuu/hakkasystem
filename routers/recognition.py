from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from ultralytics import YOLO
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import json
import tempfile
import os
import shutil
import uuid
import re

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

from database import get_db

from routers.hakka_api import (
    TextRequest,
    get_trans_token,
    call_hakka_translate_api,
    generate_hakka_tts,
    to_superscript_tone
)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

router = APIRouter(prefix="/api")

model = YOLO("yolo11n.pt")
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.8"))
GEMINI_MODE = os.getenv("GEMINI_MODE", "fallback").strip().lower()
GEMINI_REVIEW_CLASSES = {
    item.strip()
    for item in os.getenv("GEMINI_REVIEW_CLASSES", "wine glass").split(",")
    if item.strip()
}
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Gemini client（lazy init，確保容器環境變數已注入）────────────────────
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if genai is None:
        return None
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        _gemini_client = genai.Client(api_key=api_key)
        print(f"[Gemini] client 初始化成功，model={os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    except Exception as e:
        print(f"[Gemini] client 初始化失敗：{e}")
    return _gemini_client

COCO_ZH = {
    "person": "人", "bicycle": "腳踏車", "car": "汽車", "motorcycle": "摩托車",
    "airplane": "飛機", "bus": "公車", "train": "火車", "truck": "卡車",
    "boat": "船", "traffic light": "紅綠燈", "fire hydrant": "消防栓",
    "stop sign": "停止標誌", "bench": "長椅", "bird": "鳥", "cat": "貓",
    "dog": "狗", "horse": "馬", "sheep": "羊", "cow": "牛", "elephant": "大象",
    "bear": "熊", "zebra": "斑馬", "giraffe": "長頸鹿", "backpack": "背包",
    "umbrella": "雨傘", "handbag": "手提包", "tie": "領帶", "suitcase": "行李箱",
    "frisbee": "飛盤", "skis": "滑雪板", "snowboard": "滑雪板",
    "sports ball": "球", "kite": "風箏", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "衝浪板",
    "tennis racket": "網球拍", "bottle": "瓶子", "wine glass": "酒杯",
    "cup": "杯子", "fork": "叉子", "knife": "刀子", "spoon": "湯匙",
    "bowl": "碗", "banana": "香蕉", "apple": "蘋果", "sandwich": "三明治",
    "orange": "橘子", "broccoli": "花椰菜", "carrot": "紅蘿蔔",
    "hot dog": "熱狗", "pizza": "披薩", "donut": "甜甜圈", "cake": "蛋糕",
    "chair": "椅子", "couch": "沙發", "potted plant": "盆栽",
    "bed": "床", "dining table": "餐桌", "toilet": "馬桶", "tv": "電視",
    "laptop": "筆電", "mouse": "滑鼠", "remote": "遙控器", "keyboard": "鍵盤",
    "cell phone": "手機", "microwave": "微波爐", "oven": "烤箱",
    "toaster": "烤麵包機", "sink": "水槽", "refrigerator": "冰箱",
    "book": "書", "clock": "時鐘", "vase": "花瓶", "scissors": "剪刀",
    "teddy bear": "玩具熊", "hair drier": "吹風機", "toothbrush": "牙刷",
    "glasses": "眼鏡"
}


class RecognizedItem(BaseModel):
    label_en: str
    label_zh: str
    confidence: float
    source: str = "yolo"
    keyword_hakka: str = ""
    pinyin: str = ""
    audio_path: str = ""
    sentence_zh: str = ""
    sentence_hakka: str = ""
    sentence_audio_path: str = ""


class RecognitionResponse(BaseModel):
    items: list[RecognizedItem]
    image_path: str = ""


def make_numbered_text(items: list[str]) -> str:
    return "\n".join([f"{i + 1}. {item}" for i, item in enumerate(items)])


def split_numbered_text(text: str, expected_count: int) -> list[str]:
    text = text.strip()
    pattern = r"(?:\d+|[一二三四五六七八九十]+)[\.\、．]\s*"
    matches = list(re.finditer(pattern, text))

    if matches:
        parts = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parts.append(text[start:end].strip())

        while len(parts) < expected_count:
            parts.append("")

        return parts[:expected_count]

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    while len(lines) < expected_count:
        lines.append("")

    return lines[:expected_count]


def split_lines_text(text: str, expected_count: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    while len(lines) < expected_count:
        lines.append("")

    return lines[:expected_count]


def normalize_object_word(text: str) -> str:
    text = re.sub(r"[，,。.!！?？；;：:\n\r\t]", " ", text).strip()
    text = re.sub(r"\s+", " ", text)

    for prefix in ("一個", "一副", "一支", "一台", "一把", "一隻", "這是", "看起來像"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text.split(" ")[0].strip()


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _recognize_with_gemini_sync(
    image_path: str,
    yolo_candidates: list[dict] | None = None
) -> list[dict]:
    gemini_client = _get_gemini_client()
    if gemini_client is None or genai_types is None:
        return []

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    hints = ""
    if yolo_candidates:
        hints = "\nYOLO 初步辨識結果如下，這些結果可能是錯的，請只當作參考：\n"
        for item in yolo_candidates[:5]:
            hints += (
                f"- {item['label_en']} / {item['label_zh']} "
                f"(confidence={item['confidence']:.2f})\n"
            )

    prompt = f"""
你是圖片物品辨識的複查助手。
請辨識圖片中最主要、最適合拿來學習詞彙的物品單詞，最多 5 個。
每個結果只能是單一物品名詞，不要回傳句子、形容詞、場景描述、動作或用途。
繁體中文單詞請盡量控制在 2 到 6 個中文字，例如「眼鏡」、「水壺」、「鉛筆盒」。
如果 YOLO 初步結果和圖片內容不一致，請以圖片內容為準，不要照抄 YOLO。
請特別注意：
- 眼鏡、太陽眼鏡、護目鏡請回傳「眼鏡」，不要回傳「酒杯」。
- 只有真的看到有杯腳或盛酒用的玻璃杯，才可以回傳「酒杯」。
- 如果物品不確定，請選擇較通用、較適合學習的名稱。
{hints}
請只回傳 JSON，不要加解釋。

格式：
{{
  "items": [
    {{
      "label_zh": "繁體中文物品名稱",
      "label_en": "English object name",
      "confidence": 0.0
    }}
  ]
}}
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            genai_types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
            prompt,
        ],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    data = _extract_json_object(response.text or "{}")
    items = data.get("items", [])
    if not isinstance(items, list):
        return []

    cleaned_items = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        label_zh = normalize_object_word(str(item.get("label_zh", "")).strip())
        label_en = normalize_object_word(str(item.get("label_en", "")).strip())

        if not label_zh or label_zh in seen:
            continue

        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6

        cleaned_items.append({
            "label_en": label_en or label_zh,
            "label_zh": label_zh,
            "confidence": max(0.0, min(confidence, 1.0)),
            "source": "gemini",
        })
        seen.add(label_zh)

        if len(cleaned_items) >= 5:
            break

    return cleaned_items


async def recognize_with_gemini(
    image_path: str,
    yolo_candidates: list[dict] | None = None
) -> list[dict]:
    try:
        return await asyncio.to_thread(
            _recognize_with_gemini_sync,
            image_path,
            yolo_candidates
        )
    except Exception as e:
        print(f"[Gemini] 圖片辨識失敗：{e}")
        return []


async def generate_sentences_for_words(words: list[str]) -> list[str]:
    """直接呼叫 learning router 的函式，避免 HTTP loopback 造成 deadlock"""
    from routers.learning import generate_hakka_story
    from pydantic import BaseModel as _BaseModel

    class _WordsRequest(_BaseModel):
        words: list[str]

    try:
        data = await generate_hakka_story(_WordsRequest(words=words))
        sentences = []

        if isinstance(data, list):
            for i, word in enumerate(words):
                if i < len(data) and isinstance(data[i], dict):
                    sentence = (
                        data[i].get("chinese_translation")
                        or data[i].get("sentence_zh")
                        or f"這是一個{word}。"
                    )
                else:
                    sentence = f"這是一個{word}。"
                sentences.append(sentence)
            return sentences

        return [f"這是一個{word}。" for word in words]

    except Exception:
        return [f"這是一個{word}。" for word in words]


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    saved_filename = f"{uuid.uuid4().hex}{suffix}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    try:
        shutil.copy(tmp_path, saved_path)

        results = model(tmp_path, verbose=False)
        boxes = results[0].boxes

        yolo_candidates = []
        if boxes is not None and len(boxes) > 0:
            sorted_indices = boxes.conf.argsort(descending=True)

            for idx in sorted_indices:
                i = int(idx)
                label_en = model.names[int(boxes.cls[i])]
                confidence = float(boxes.conf[i])

                if label_en not in [obj["label_en"] for obj in yolo_candidates]:
                    yolo_candidates.append({
                        "label_en": label_en,
                        "label_zh": COCO_ZH.get(label_en, label_en),
                        "confidence": confidence,
                        "source": "yolo"
                    })

                if len(yolo_candidates) >= 5:
                    break

        detected_objects = []
        gemini_reason = ""
        should_use_gemini = GEMINI_MODE in {"always", "force", "gemini"}

        if should_use_gemini:
            gemini_reason = f"GEMINI_MODE={GEMINI_MODE}"

        if not should_use_gemini and (boxes is None or len(boxes) == 0):
            should_use_gemini = True
            gemini_reason = "YOLO 未偵測到物件"

        if not should_use_gemini:
            max_confidence = yolo_candidates[0]["confidence"]
            should_use_gemini = max_confidence < YOLO_CONF_THRESHOLD
            if should_use_gemini:
                gemini_reason = (
                    f"YOLO 最高信心度 {max_confidence:.2f} "
                    f"低於門檻 {YOLO_CONF_THRESHOLD:.2f}"
                )

        if not should_use_gemini and GEMINI_REVIEW_CLASSES:
            top_label_en = yolo_candidates[0]["label_en"]
            if top_label_en in GEMINI_REVIEW_CLASSES:
                should_use_gemini = True
                gemini_reason = f"YOLO 類別 {top_label_en} 設定為需要 Gemini 複查"

        if should_use_gemini:
            print(f"[Recognition] 使用 Gemini 輔助辨識：{gemini_reason}")
            detected_objects = await recognize_with_gemini(
                tmp_path,
                yolo_candidates
            )
            if not detected_objects:
                print("[Recognition] Gemini 未回傳物品單詞")

        if (
            not detected_objects
            and boxes is not None
            and len(boxes) > 0
        ):
            if should_use_gemini:
                print("[Recognition] Gemini 無結果，暫時回退使用 YOLO 候選")
            detected_objects = yolo_candidates

        if not detected_objects:
            if _get_gemini_client() is None:
                raise HTTPException(
                    status_code=422,
                    detail="未偵測到任何物件，且尚未設定 GEMINI_API_KEY"
                )

            raise HTTPException(
                status_code=422,
                detail="未偵測到任何物件，Gemini 也沒有回傳可用單詞"
            )

        words_zh = [obj["label_zh"] for obj in detected_objects]

        # 取得翻譯 token（手動呼叫，不透過 FastAPI Depends）
        trans_token = await get_trans_token()

        # 1. 中文單字 → 客語：分開翻譯，比較不會變成句子用法
        hakka_words = []

        for word_zh in words_zh:
            hakka_result = await call_hakka_translate_api(
                endpoint="/MT/translate/hakka_zh_hk",
                text=word_zh,
                token=trans_token
            )

            keyword_hakka = hakka_result.get("output", "") or word_zh
            hakka_words.append(keyword_hakka)

        # 2. 客語單字 → 拼音：批次處理
        hakka_lines = "\n".join(hakka_words)

        batch_pinyin_result = await call_hakka_translate_api(
            endpoint="/MT/translate/hakka_hk_py",
            text=hakka_lines,
            token=trans_token
        )

        pinyin_words = split_lines_text(
            batch_pinyin_result.get("output", ""),
            expected_count=len(words_zh)
        )
        pinyin_words = [to_superscript_tone(p) for p in pinyin_words]

        # 3. LLM 批次造句：每個詞各自一句
        sentences_zh = await generate_sentences_for_words(words_zh)

        # 4. 中文句子 → 客語句子：批次翻譯
        sentences_numbered = make_numbered_text(sentences_zh)

        batch_sentence_hakka_result = await call_hakka_translate_api(
            endpoint="/MT/translate/hakka_zh_hk",
            text=sentences_numbered,
            token=trans_token
        )

        sentences_hakka = split_numbered_text(
            batch_sentence_hakka_result.get("output", ""),
            expected_count=len(words_zh)
        )

        items = []

        # 5. TTS 分開產生，因為每個單字與句子都要獨立 wav
        for i, obj in enumerate(detected_objects):
            word_zh = words_zh[i]
            keyword_hakka = hakka_words[i] if i < len(hakka_words) else word_zh
            pinyin = pinyin_words[i] if i < len(pinyin_words) else ""

            sentence_zh = (
                sentences_zh[i]
                if i < len(sentences_zh)
                else f"這是一個{word_zh}。"
            )

            sentence_hakka = (
                sentences_hakka[i]
                if i < len(sentences_hakka)
                else ""
            )

            audio_path = await generate_hakka_tts(
                keyword_hakka,
                folder="words"
            )

            sentence_audio_path = await generate_hakka_tts(
                sentence_hakka,
                folder="sentences"
            )

            items.append(
                RecognizedItem(
                    label_en=obj["label_en"],
                    label_zh=word_zh,
                    confidence=round(obj["confidence"], 2),
                    source=obj.get("source", "yolo"),
                    keyword_hakka=keyword_hakka,
                    pinyin=pinyin,
                    audio_path=audio_path,
                    sentence_zh=sentence_zh,
                    sentence_hakka=sentence_hakka,
                    sentence_audio_path=sentence_audio_path
                )
            )

        return RecognitionResponse(
            items=items,
            image_path="/" + saved_path.replace("\\", "/")
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
