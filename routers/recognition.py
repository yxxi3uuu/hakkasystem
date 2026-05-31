from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from ultralytics import YOLO
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import os
import shutil
import uuid
import re

from database import get_db

from routers.hakka_api import (
    TextRequest,
    get_trans_token,
    call_hakka_translate_api,
    generate_hakka_tts,
    to_superscript_tone
)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

router = APIRouter(prefix="/api")

model = YOLO("yolo11n.pt")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

        if boxes is None or len(boxes) == 0:
            raise HTTPException(status_code=422, detail="未偵測到任何物件")

        detected_objects = []
        sorted_indices = boxes.conf.argsort(descending=True)

        for idx in sorted_indices:
            i = int(idx)
            label_en = model.names[int(boxes.cls[i])]
            confidence = float(boxes.conf[i])

            if label_en not in [obj["label_en"] for obj in detected_objects]:
                detected_objects.append({
                    "label_en": label_en,
                    "label_zh": COCO_ZH.get(label_en, label_en),
                    "confidence": confidence
                })

            if len(detected_objects) >= 5:
                break

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