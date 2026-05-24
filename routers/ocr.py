"""
雙軌 OCR Router
- 第一軌：Tesseract（離線，低延遲，適合印刷體中文）
- 第二軌：PaddleOCR（後端，高精度，適合複雜背景/手寫/反光）
流程：先跑 Tesseract，若結果為空或字數過少，自動 fallback 到 PaddleOCR
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import os
import re

router = APIRouter(prefix="/api/ocr", tags=["OCR"])

# ── 延遲載入，避免啟動時因套件未安裝而崩潰 ──
_tesseract_ok = False
_paddle_ocr = None


def _init_tesseract():
    global _tesseract_ok
    try:
        import pytesseract
        # Windows 預設安裝路徑，若不同請修改
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                pytesseract.pytesseract.tesseract_cmd = c
                break
        # 測試是否可用
        pytesseract.get_tesseract_version()
        _tesseract_ok = True
    except Exception as e:
        print(f"[OCR] Tesseract 初始化失敗（將只用 PaddleOCR）: {e}")
        _tesseract_ok = False


def _init_paddle():
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    try:
        from paddleocr import PaddleOCR
        # use_angle_cls=True 支援旋轉文字；lang='ch' 支援繁簡中文
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        print("[OCR] PaddleOCR 初始化成功")
    except Exception as e:
        print(f"[OCR] PaddleOCR 初始化失敗: {e}")
        _paddle_ocr = None
    return _paddle_ocr


# 啟動時嘗試初始化
_init_tesseract()


# ── 工具函式 ──

def _clean_text(text: str) -> str:
    """移除雜訊字元，只保留中文、英文、數字、常用標點"""
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf"
                  r"a-zA-Z0-9\s，。！？、：；「」『』（）\(\)\-]", "", text)
    return text.strip()


def _run_tesseract(image_path: str) -> str:
    """用 Tesseract 跑 OCR，回傳清理後的文字"""
    if not _tesseract_ok:
        print("[OCR] Tesseract 不可用，跳過")
        return ""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        raw = pytesseract.image_to_string(img, lang="chi_tra+chi_sim+eng")
        cleaned = _clean_text(raw)
        print(f"[OCR] Tesseract 原始結果: {repr(raw[:100])}")
        print(f"[OCR] Tesseract 清理後: {repr(cleaned[:100])}")
        return cleaned
    except Exception as e:
        print(f"[OCR] Tesseract 執行失敗: {e}")
        return ""


def _run_paddle(image_path: str) -> str:
    """用 PaddleOCR 跑 OCR，回傳清理後的文字"""
    ocr = _init_paddle()
    if ocr is None:
        print("[OCR] PaddleOCR 不可用，跳過")
        return ""
    try:
        result = ocr.ocr(image_path, cls=True)
        print(f"[OCR] PaddleOCR 原始結果: {result}")
        lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text, conf = line[1]
                    print(f"[OCR] PaddleOCR 行: {repr(text)} 信心度={conf:.2f}")
                    if conf >= 0.5:
                        lines.append(text)
        cleaned = _clean_text(" ".join(lines))
        print(f"[OCR] PaddleOCR 清理後: {repr(cleaned[:100])}")
        return cleaned
    except Exception as e:
        print(f"[OCR] PaddleOCR 執行失敗: {e}")
        return ""


# ── Response Schema ──

class OcrResult(BaseModel):
    text: str                  # 辨識出的文字（已清理）
    engine: str                # 使用的引擎：tesseract / paddle / none
    char_count: int            # 有效字元數


class OcrWithHakkaResult(BaseModel):
    text: str
    engine: str
    char_count: int
    hakka: str = ""            # 客語翻譯
    audio_path: str = ""       # 客語 TTS 音檔路徑


# ── Endpoints ──

@router.post("/extract", response_model=OcrResult)
async def extract_text(file: UploadFile = File(...)):
    """
    純 OCR：上傳圖片，回傳辨識文字。
    雙軌策略：Tesseract 優先，結果不足時 fallback PaddleOCR。
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 第一軌：Tesseract（快速離線）
        text = _run_tesseract(tmp_path)
        engine = "tesseract"

        # 若 Tesseract 結果少於 2 個中文字，切換到 PaddleOCR
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(zh_chars) < 2:
            paddle_text = _run_paddle(tmp_path)
            if paddle_text:
                text = paddle_text
                engine = "paddle"

        if not text:
            engine = "none"

        return OcrResult(
            text=text,
            engine=engine,
            char_count=len(text.replace(" ", ""))
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/extract-and-translate", response_model=OcrWithHakkaResult)
async def extract_and_translate(file: UploadFile = File(...)):
    """
    OCR + 客語翻譯 + TTS：上傳圖片，辨識文字後直接翻成客語並產生音檔。
    """
    from routers.hakka_api import get_trans_token, call_hakka_translate_api, generate_hakka_tts

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # ── OCR ──
        text = _run_tesseract(tmp_path)
        engine = "tesseract"

        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(zh_chars) < 2:
            paddle_text = _run_paddle(tmp_path)
            if paddle_text:
                text = paddle_text
                engine = "paddle"

        if not text:
            return OcrWithHakkaResult(
                text="", engine="none", char_count=0,
                hakka="", audio_path=""
            )

        # 截斷過長文字（API 限制 300 字）
        text_for_api = text[:300]

        # ── 客語翻譯 ──
        hakka = ""
        audio_path = ""
        try:
            token = await get_trans_token()
            result = await call_hakka_translate_api(
                endpoint="/MT/translate/hakka_zh_hk",
                text=text_for_api,
                token=token
            )
            hakka = result.get("output", "") or ""

            # ── TTS ──
            if hakka:
                audio_path = await generate_hakka_tts(hakka[:200], folder="ocr")
        except Exception as e:
            print(f"[OCR] 翻譯/TTS 失敗: {e}")

        return OcrWithHakkaResult(
            text=text,
            engine=engine,
            char_count=len(text.replace(" ", "")),
            hakka=hakka,
            audio_path=audio_path
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
