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
    "teddy bear": "玩具熊", "hair drier": "吹風機", "toothbrush": "牙刷","glasses":"眼鏡"
}

class RecognizedItem(BaseModel):
    label_en: str
    label_zh: str
    confidence: float
    hakka_sentence: str = ""
    chinese_translation: str = ""

class RecognitionResponse(BaseModel):
    items: list[RecognizedItem]

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

        # 取得不重複的 top 5 物件 (盡量非人)
        detected_objects = []
        # 先以信心度排序
        sorted_indices = boxes.conf.argsort(descending=True)
        
        for idx in sorted_indices:
            i = int(idx)
            lbl = model.names[int(boxes.cls[i])]
            conf = float(boxes.conf[i])
            if lbl not in [obj["label_en"] for obj in detected_objects]:
                detected_objects.append({
                    "label_en": lbl,
                    "confidence": conf
                })
            if len(detected_objects) >= 5:
                break
                
        # 準備請求的 words
        words = []
        for obj in detected_objects:
            lbl_zh = COCO_ZH.get(obj["label_en"], obj["label_en"])
            obj["label_zh"] = lbl_zh
            words.append(lbl_zh)
            
        # 呼叫 LLM 生成客語多句情境
        story_result = []
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "http://localhost:8000/api/learning/generate-story",
                    json={"words": words}
                )
                if resp.status_code == 200:
                    story_result = resp.json()
        except Exception as e:
            print(f"Error calling story generation: {e}")
            pass
            
        # 組合結果
        items = []
        for obj in detected_objects:
            hakka_sentence = ""
            chinese_translation = ""
            # 尋找對應的句子
            for s in story_result:
                if s.get("word") == obj["label_zh"]:
                    hakka_sentence = s.get("hakka_sentence", "")
                    chinese_translation = s.get("chinese_translation", "")
                    break
                    
            items.append(RecognizedItem(
                label_en=obj["label_en"],
                label_zh=obj["label_zh"],
                confidence=round(obj["confidence"], 2),
                hakka_sentence=hakka_sentence,
                chinese_translation=chinese_translation
            ))

        return RecognitionResponse(items=items)
    finally:
        os.unlink(tmp_path)
