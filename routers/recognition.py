from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import httpx
import tempfile
import os

router = APIRouter(prefix="/api")

# 載入 YOLO 模型（yolo11n 最輕量，首次執行會自動下載）
model = YOLO("yolo11n.pt")

# 中文類別名稱對照（COCO 80 類）
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
}

class RecognitionResponse(BaseModel):
    label_en: str
    label_zh: str
    confidence: float
    hakka_sentence: str = ""
    chinese_translation: str = ""

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(file: UploadFile = File(...)):
    # 儲存上傳的圖片到暫存檔
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # YOLO 推論
        results = model(tmp_path, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            raise HTTPException(status_code=422, detail="未偵測到任何物件")

        # 取信心值最高的結果
        best_idx = int(boxes.conf.argmax())
        label_en = model.names[int(boxes.cls[best_idx])]
        confidence = float(boxes.conf[best_idx])
        label_zh = COCO_ZH.get(label_en, label_en)

        # 呼叫 LLM 生成客語例句
        hakka_sentence = ""
        chinese_translation = ""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "http://localhost:8000/api/learning/generate-sentence",
                    json={"word": label_zh}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    hakka_sentence = data.get("hakka_sentence", "")
                    chinese_translation = data.get("chinese_translation", "")
        except Exception:
            pass  # LLM 失敗不影響辨識結果

        return RecognitionResponse(
            label_en=label_en,
            label_zh=label_zh,
            confidence=round(confidence, 2),
            hakka_sentence=hakka_sentence,
            chinese_translation=chinese_translation,
        )
    finally:
        os.unlink(tmp_path)
