from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
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

# ── YOLO 延遲載入：只在 Gemini 無法使用或辨識失敗時才初始化 ──────────────
_yolo_model = None

def _get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    print("[YOLO] 初次載入模型 yolo11n.pt ...")
    from ultralytics import YOLO as _YOLO
    _yolo_model = _YOLO("yolo11n.pt")
    print("[YOLO] 模型載入完成")
    return _yolo_model

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
CROP_DIR = os.path.join(UPLOAD_DIR, "crops")
os.makedirs(CROP_DIR, exist_ok=True)


def _refine_bbox_for_label(
    label_zh: str,
    bbox: list[float],
    image_width: int,
    image_height: int
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = x2 - x1
    box_h = y2 - y1

    if "眼鏡" in label_zh and box_w > 0 and box_h > 0:
        # Gemini often frames the whole upper face for glasses. Keep the lower
        # middle band where the frame/lenses usually sit.
        if box_h > image_height * 0.07:
            y1 = y1 + box_h * 0.42
            y2 = y2 - box_h * 0.06

        if box_w / max(box_h, 1) < 2.0:
            y1 = y1 + box_h * 0.15
            y2 = y2 - box_h * 0.10

    x1 = max(0.0, min(x1, image_width))
    x2 = max(0.0, min(x2, image_width))
    y1 = max(0.0, min(y1, image_height))
    y2 = max(0.0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return [float(v) for v in bbox]

    return [x1, y1, x2, y2]


def _save_object_crop(image_path: str, bbox: list[float], label_zh: str = "") -> tuple[list[float], str]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
            x1, y1, x2, y2 = _refine_bbox_for_label(label_zh, bbox, width, height)
            pad_ratio = 0.01 if "眼鏡" in label_zh else 0.03
            pad_x = (x2 - x1) * pad_ratio
            pad_y = (y2 - y1) * pad_ratio

            left = max(0, int(x1 - pad_x))
            top = max(0, int(y1 - pad_y))
            right = min(width, int(x2 + pad_x))
            bottom = min(height, int(y2 + pad_y))

            if right <= left or bottom <= top:
                return [x1, y1, x2, y2], ""

            crop = image.crop((left, top, right, bottom))
            crop_filename = f"{uuid.uuid4().hex}.jpg"
            crop_path = os.path.join(CROP_DIR, crop_filename)
            crop.convert("RGB").save(crop_path, "JPEG", quality=90)

            return [left, top, right, bottom], "/" + crop_path.replace("\\", "/")
    except Exception as e:
        print(f"[Recognition] crop failed: {e}")
        return [float(v) for v in bbox], ""

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
    "person": "人物", "bicycle": "腳踏車", "car": "汽車", "motorcycle": "摩托車",
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
    bbox: list[float] | None = None
    crop_path: str = ""
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


def normalize_person_label(label_zh: str, label_en: str = "") -> str:
    text = label_zh.strip()
    en = label_en.strip().lower()

    male_words = {"男", "男生", "男人", "男性", "男孩", "男童", "先生"}
    female_words = {"女", "女生", "女人", "女性", "女孩", "女童", "小姐"}

    if text in male_words or en in {"male", "man", "boy", "gentleman"}:
        return "男生"
    if text in female_words or en in {"female", "woman", "girl", "lady"}:
        return "女生"
    if text in {"人", "人物", "人像", "行人", "person", "people"} or en in {"person", "people", "human"}:
        return "人物"

    return text


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


def _bbox_from_gemini_item(item: dict, image_width: int, image_height: int) -> list[float] | None:
    raw_bbox = item.get("bbox_xyxy")
    is_normalized = raw_bbox is not None

    if raw_bbox is None:
        raw_bbox = (
            item.get("bbox")
            or item.get("box")
            or item.get("bounding_box")
        )
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return None

    # Gemini is prompted to return bbox_xyxy normalized to 0-1000.
    if is_normalized or max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1:
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1:
            x1 *= 1000
            x2 *= 1000
            y1 *= 1000
            y2 *= 1000
        x1 = x1 / 1000 * image_width
        x2 = x2 / 1000 * image_width
        y1 = y1 / 1000 * image_height
        y2 = y2 / 1000 * image_height

    x1 = max(0.0, min(x1, image_width))
    x2 = max(0.0, min(x2, image_width))
    y1 = max(0.0, min(y1, image_height))
    y2 = max(0.0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def _is_plausible_bbox_for_label(label_zh: str, bbox: list[float] | None, image_width: int, image_height: int) -> bool:
    if not bbox:
        return False

    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return False

    if "眼鏡" in label_zh:
        aspect = box_w / box_h
        center_y = y1 + box_h / 2
        return (
            1.2 <= aspect <= 5.0
            and box_w >= image_width * 0.06
            and box_h >= image_height * 0.025
            and center_y >= image_height * 0.14
            and y2 <= image_height * 0.72
        )

    return True


def _recognize_with_gemini_sync(
    image_path: str,
    yolo_candidates: list[dict] | None = None
) -> list[dict]:
    gemini_client = _get_gemini_client()
    if gemini_client is None or genai_types is None:
        return []

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    from PIL import Image
    with Image.open(image_path) as img:
        image_width, image_height = img.size

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
      "confidence": 0.0,
      "bbox_xyxy": [0, 0, 1000, 1000]
    }}
  ]
}}

Important:
- Return separate items for separate visible objects, including small accessories such as glasses, hats, cups, phones, bags, and watches.
- For each item, include bbox_xyxy as [x1, y1, x2, y2] normalized to 0-1000 over the full image.
- The bounding box must tightly focus on that specific object. If a person is wearing glasses, return one item for the person and another item for the glasses with different boxes.
- Only return glasses if the eyeglass frame or lenses are clearly visible. The glasses box must enclose the frame/lenses, not forehead, hair, cheek, or background.
- Do not return the Chinese label "人". For a visible person, return "男生" or "女生" when reasonably inferable; if gender is unclear, return "人物".
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
        label_zh = normalize_person_label(label_zh, label_en)

        if not label_zh or label_zh in seen:
            continue

        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6

        bbox = _bbox_from_gemini_item(item, image_width, image_height)
        if "眼鏡" in label_zh and not _is_plausible_bbox_for_label(label_zh, bbox, image_width, image_height):
            print(f"[Gemini] skip implausible glasses bbox: {bbox}")
            continue

        crop_path = ""
        if bbox:
            bbox, crop_path = _save_object_crop(image_path, bbox, label_zh)

        cleaned_items.append({
            "label_en": label_en or label_zh,
            "label_zh": label_zh,
            "confidence": max(0.0, min(confidence, 1.0)),
            "source": "gemini",
            "bbox": bbox,
            "crop_path": crop_path,
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


def attach_yolo_bboxes(
    detected_objects: list[dict],
    yolo_candidates: list[dict]
) -> list[dict]:
    if not detected_objects or not yolo_candidates:
        return detected_objects

    used_candidate_indices = set()

    for obj_index, obj in enumerate(detected_objects):
        if obj.get("bbox"):
            continue

        label_en = str(obj.get("label_en", "")).strip().lower()
        label_zh = str(obj.get("label_zh", "")).strip().lower()
        matched_index = None

        for candidate_index, candidate in enumerate(yolo_candidates):
            if candidate_index in used_candidate_indices:
                continue

            candidate_en = str(candidate.get("label_en", "")).strip().lower()
            candidate_zh = str(candidate.get("label_zh", "")).strip().lower()
            if label_en and label_en == candidate_en:
                matched_index = candidate_index
                break
            if label_zh and label_zh == candidate_zh:
                matched_index = candidate_index
                break

        if matched_index is None and obj_index < len(yolo_candidates):
            matched_index = obj_index

        if matched_index is not None:
            bbox = yolo_candidates[matched_index].get("bbox")
            if bbox:
                obj["bbox"] = bbox
                obj["crop_path"] = yolo_candidates[matched_index].get("crop_path", "")
                used_candidate_indices.add(matched_index)

    return detected_objects


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

        detected_objects = []
        yolo_candidates = []

        # ── 決策：優先用 Gemini，YOLO 只在 Gemini 不可用或無結果時才載入 ──
        gemini_client_available = _get_gemini_client() is not None
        force_yolo = GEMINI_MODE in {"yolo", "disabled"}

        if not force_yolo and gemini_client_available:
            # 第一軌：直接讓 Gemini 辨識，不預先跑 YOLO
            print("[Recognition] 使用 Gemini 辨識（跳過 YOLO 初始載入）")
            detected_objects = await recognize_with_gemini(tmp_path, yolo_candidates=None)
            if not detected_objects:
                print("[Recognition] Gemini 未回傳物品，fallback 到 YOLO")

        if not detected_objects:
            # 第二軌（fallback）：Gemini 無結果 或 未設定 Gemini → 載入 YOLO
            yolo_model = _get_yolo_model()
            results = yolo_model(tmp_path, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                sorted_indices = boxes.conf.argsort(descending=True)

                for idx in sorted_indices:
                    i = int(idx)
                    label_en = yolo_model.names[int(boxes.cls[i])]
                    confidence = float(boxes.conf[i])

                    if label_en not in [obj["label_en"] for obj in yolo_candidates]:
                        raw_bbox = [float(v) for v in boxes.xyxy[i].tolist()]
                        label_zh = COCO_ZH.get(label_en, label_en)
                        bbox, crop_path = _save_object_crop(tmp_path, raw_bbox, label_zh)
                        yolo_candidates.append({
                            "label_en": label_en,
                            "label_zh": label_zh,
                            "confidence": confidence,
                            "source": "yolo",
                            "bbox": bbox,
                            "crop_path": crop_path,
                        })

                    if len(yolo_candidates) >= 5:
                        break

            # YOLO 有結果但信心度低 → 再試一次 Gemini（帶 YOLO hints）
            if yolo_candidates and gemini_client_available and not force_yolo:
                max_confidence = yolo_candidates[0]["confidence"]
                top_label_en = yolo_candidates[0]["label_en"]
                needs_gemini_review = (
                    max_confidence < YOLO_CONF_THRESHOLD
                    or top_label_en in GEMINI_REVIEW_CLASSES
                )
                if needs_gemini_review:
                    print(f"[Recognition] YOLO 信心度 {max_confidence:.2f}，帶 hints 再問 Gemini")
                    gemini_result = await recognize_with_gemini(tmp_path, yolo_candidates)
                    if gemini_result:
                        detected_objects = attach_yolo_bboxes(gemini_result, yolo_candidates)
                    else:
                        detected_objects = yolo_candidates
                else:
                    detected_objects = yolo_candidates
            elif yolo_candidates:
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

        for obj in detected_objects:
            obj["label_zh"] = normalize_person_label(
                str(obj.get("label_zh", "")),
                str(obj.get("label_en", ""))
            )

        words_zh = [obj["label_zh"] for obj in detected_objects]

        # 取得翻譯 token（手動呼叫，不透過 FastAPI Depends）
        trans_token = await get_trans_token()

        # ── 全部翻譯 / 造句 / TTS 同步並行，大幅縮短等待時間 ──────────────

        # 1. 中文單字 → 客語（並行）
        hakka_word_results = await asyncio.gather(*[
            call_hakka_translate_api(
                endpoint="/MT/translate/hakka_zh_hk",
                text=word_zh,
                token=trans_token
            )
            for word_zh in words_zh
        ])
        hakka_words = [
            r.get("output", "") or words_zh[i]
            for i, r in enumerate(hakka_word_results)
        ]

        # 2. 客語單字 → 拼音（並行）
        async def _safe_pinyin(hakka: str) -> str:
            try:
                result = await call_hakka_translate_api(
                    endpoint="/MT/translate/hakka_hk_py",
                    text=hakka,
                    token=trans_token
                )
                return to_superscript_tone(result.get("output", "").strip())
            except Exception as e:
                print(f"[Recognition] 拼音轉換失敗：{hakka} / {e}")
                return ""

        # 3. LLM 批次造句（與拼音並行）
        pinyin_task = asyncio.gather(*[_safe_pinyin(h) for h in hakka_words])
        sentences_task = generate_sentences_for_words(words_zh)

        pinyin_words, sentences_zh = await asyncio.gather(pinyin_task, sentences_task)
        pinyin_words = list(pinyin_words)

        # 4. 中文句子 → 客語句子（批次翻譯，一次 API）
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

        # 5. TTS 分批並行（Semaphore 限制同時最多 3 個，避免 rate limit）
        _tts_semaphore = asyncio.Semaphore(3)

        async def _safe_tts(text: str, folder: str) -> str:
            async with _tts_semaphore:
                try:
                    return await generate_hakka_tts(text, folder=folder)
                except Exception as e:
                    print(f"[Recognition] TTS 失敗：{text[:20]} / {e}")
                    return ""

        tts_tasks = []
        for i in range(len(detected_objects)):
            keyword_hakka = hakka_words[i] if i < len(hakka_words) else words_zh[i]
            sentence_hakka = sentences_hakka[i] if i < len(sentences_hakka) else ""
            tts_tasks.append(_safe_tts(keyword_hakka, "words"))
            tts_tasks.append(_safe_tts(sentence_hakka, "sentences"))

        tts_results = await asyncio.gather(*tts_tasks)

        items = []
        for i, obj in enumerate(detected_objects):
            word_zh = words_zh[i]
            keyword_hakka = hakka_words[i] if i < len(hakka_words) else word_zh
            pinyin = pinyin_words[i] if i < len(pinyin_words) else ""
            sentence_zh = sentences_zh[i] if i < len(sentences_zh) else f"這是一個{word_zh}。"
            sentence_hakka = sentences_hakka[i] if i < len(sentences_hakka) else ""
            audio_path = tts_results[i * 2]
            sentence_audio_path = tts_results[i * 2 + 1]

            items.append(
                RecognizedItem(
                    label_en=obj["label_en"],
                    label_zh=word_zh,
                    confidence=round(obj["confidence"], 2),
                    source=obj.get("source", "yolo"),
                    bbox=obj.get("bbox"),
                    crop_path=obj.get("crop_path", ""),
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


