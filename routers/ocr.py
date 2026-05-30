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


def _run_tesseract(image_path: str) -> tuple[str, float]:
    """
    用 Tesseract 跑 OCR，回傳 (清理後的文字, 平均信心度 0~100)。
    信心度低代表辨識品質差，應 fallback 到 PaddleOCR。
    """
    if not _tesseract_ok:
        print("[OCR] Tesseract 不可用，跳過")
        return "", 0.0
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)

        # image_to_data 可取得每個字的信心度
        data = pytesseract.image_to_data(
            img,
            lang="chi_tra+chi_sim+eng",
            output_type=pytesseract.Output.DICT
        )

        words  = []
        confs  = []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            conf = int(conf)
            if text and conf > 0:   # conf == -1 表示非文字區塊
                words.append(text)
                confs.append(conf)

        raw     = " ".join(words)
        cleaned = _clean_text(raw)
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0

        print(f"[OCR] Tesseract 清理後: {repr(cleaned[:100])}  平均信心度: {avg_conf:.1f}")
        return cleaned, avg_conf

    except Exception as e:
        print(f"[OCR] Tesseract 執行失敗: {e}")
        return "", 0.0


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


class KeywordItem(BaseModel):
    word_zh: str               # 中文詞彙
    word_hakka: str = ""       # 客語漢字翻譯
    word_pinyin: str = ""      # 客語拼音（四縣腔）
    audio_path: str = ""       # 客語 TTS 音檔


class OcrWithKeywordsResult(BaseModel):
    text: str                  # 完整辨識文字（中文）
    engine: str
    char_count: int
    full_hakka: str = ""       # 全文客語翻譯
    full_audio_path: str = ""  # 全文客語 TTS 音檔
    keywords: list[KeywordItem] = []   # 抽出的關鍵詞彙清單


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
        text, conf = _run_tesseract(tmp_path)
        engine = "tesseract"

        # fallback 條件：中文字太少，或平均信心度低於 60（代表辨識品質差）
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(zh_chars) < 2 or conf < 60:
            print(f"[OCR] Tesseract 信心度 {conf:.1f} 不足，切換 PaddleOCR")
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
        text, conf = _run_tesseract(tmp_path)
        engine = "tesseract"

        # fallback 條件：中文字太少，或平均信心度低於 60
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(zh_chars) < 2 or conf < 60:
            print(f"[OCR] Tesseract 信心度 {conf:.1f} 不足，切換 PaddleOCR")
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


# ── LLM 關鍵詞抽取 ──────────────────────────────────────────────────────────

import os as _os
import httpx as _httpx

_OPENAI_API_KEY  = _os.getenv("OPENAI_API_KEY", "")
_OPENAI_API_BASE = _os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
_OPENAI_MODEL    = _os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_KEYWORD_PROMPT = """你是一位中文語言教學專家，同時也是嚴格的文字品管員。
以下是一段從圖片 OCR 辨識出的中文文字（可能含有辨識錯誤）：

「{text}」

你的任務：
1. 先判斷這段文字是否為「可讀的正常中文」。如果文字明顯是亂碼或辨識錯誤（例如「三注音妨礙外」這種無意義組合），請直接回傳空陣列 []。
2. 如果文字正常，從中抽出 5 到 8 個「真實存在於中文詞典的詞彙」（名詞、動詞、形容詞優先，長度 2~4 字，避免虛詞、標點、數字）。
3. 每個詞彙必須是完整、有意義的中文詞，不可以是隨機切割的字組合。

只回傳一個 JSON 陣列，不要有任何額外說明：
["詞彙1", "詞彙2", "詞彙3"]"""


async def _extract_keywords_llm(text: str) -> list[str]:
    """呼叫 LLM 從文章中抽出關鍵詞彙，失敗時 fallback 用簡單斷詞。"""
    if not _OPENAI_API_KEY:
        print("[OCR] OPENAI_API_KEY 未設定，使用簡單斷詞 fallback")
        return _simple_keyword_fallback(text)

    prompt = _KEYWORD_PROMPT.format(text=text[:800])  # 最多送 800 字給 LLM
    try:
        async with _httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{_OPENAI_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {_OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 200,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # 解析 JSON 陣列
            import json as _json
            # 容錯：有時 LLM 會在 JSON 前後加說明文字，用 regex 抓出陣列
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                keywords = _json.loads(match.group())
                # 過濾空字串，最多取 8 個
                return [k.strip() for k in keywords if k.strip()][:8]
    except Exception as e:
        print(f"[OCR] LLM 關鍵詞抽取失敗：{e}，使用 fallback")

    return _simple_keyword_fallback(text)


def _simple_keyword_fallback(text: str) -> list[str]:
    """
    無 LLM / jieba 時的 fallback：
    用常見中文詞彙規則從文章中抽取有意義的詞彙。
    優先順序：jieba → 規則抽取 → 空陣列
    """
    # 先做基本品質檢查：中文字少於 4 個，直接放棄
    zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(zh_chars) < 4:
        return []

    # ── 方法一：jieba 斷詞（若有安裝）──────────────────────────────────
    try:
        import jieba
        import jieba.posseg as pseg

        words_flags = list(pseg.cut(text))
        candidates = [
            w.word for w, flag in words_flags
            if flag.startswith(('n', 'v', 'a'))
            and 2 <= len(w.word) <= 4
            and re.fullmatch(r'[\u4e00-\u9fff]+', w.word)
        ]
        seen: set[str] = set()
        result = []
        for w in candidates:
            if w not in seen:
                seen.add(w)
                result.append(w)
        if result:
            return result[:8]
    except Exception:
        pass

    # ── 方法二：規則抽取（不需要任何套件）──────────────────────────────
    # 策略：抽取文章中連續出現的 2~4 個中文字組合，
    # 過濾掉單純的虛詞、代詞、數字詞
    STOP_WORDS = {
        '的', '了', '在', '是', '我', '他', '她', '它', '你', '們',
        '這', '那', '有', '也', '都', '就', '不', '很', '一', '個',
        '上', '下', '來', '去', '到', '和', '與', '或', '但', '如',
        '果', '因', '為', '所', '以', '而', '且', '雖', '然', '雖然',
        '如果', '因為', '所以', '而且', '一定', '一個', '一隻', '一些',
        '小朋友', '時候', '時節',  # 保留這些常見詞
    }
    # 保留清單（強制納入的高價值詞彙）
    KEEP_WORDS = {
        '小朋友', '習慣', '學習', '拼音', '符號', '注音', '外文',
        '家長', '年級', '班上', '起跑', '輸入', '規則',
    }

    # 先從保留清單找
    found = [w for w in KEEP_WORDS if w in text]

    # 再用 regex 抓 2~4 字的純中文詞組（排除停用詞）
    pattern = re.compile(r'[\u4e00-\u9fff]{2,4}')
    all_matches = pattern.findall(text)

    seen2: set[str] = set(found)
    for w in all_matches:
        if w not in seen2 and w not in STOP_WORDS and len(w) >= 2:
            # 簡單品質過濾：不能全是同一個字重複（如「一一」）
            if len(set(w)) > 1:
                seen2.add(w)
                found.append(w)
        if len(found) >= 8:
            break

    return found[:8]


@router.post("/extract-and-keywords", response_model=OcrWithKeywordsResult,
             summary="OCR + 全文翻譯 + LLM 關鍵詞抽取 + 客語翻譯")
async def extract_and_keywords(file: UploadFile = File(...)):
    """
    長文章完整流程：
    1. OCR 辨識全文
    2. 全文翻成客語 + 生成全文 TTS（讓使用者聽整篇）
    3. LLM 自動抽出 5~8 個關鍵詞彙
    4. 每個詞彙各自翻成客語 + 生成 TTS（讓使用者逐一選擇儲存）
    """
    import asyncio
    from routers.hakka_api import get_trans_token, call_hakka_translate_api, generate_hakka_tts

    suffix = _os.path.splitext(file.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # ── 1. OCR ──────────────────────────────────────────────────────────
        text, conf = _run_tesseract(tmp_path)
        engine = "tesseract"

        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(zh_chars) < 2 or conf < 60:
            print(f"[OCR] 信心度 {conf:.1f} 不足，切換 PaddleOCR")
            paddle_text = _run_paddle(tmp_path)
            if paddle_text:
                text = paddle_text
                engine = "paddle"

        if not text:
            return OcrWithKeywordsResult(
                text="", engine="none", char_count=0, keywords=[]
            )

        # ── 2. 全文翻譯 + 全文 TTS（與關鍵詞抽取並行）──────────────────────
        async def _full_translate() -> tuple[str, str]:
            """全文中文 → 客語翻譯 + TTS，回傳 (hakka_text, audio_path)"""
            try:
                token = await get_trans_token()
                # 翻譯 API 限制 300 字，超過截斷
                result = await call_hakka_translate_api(
                    endpoint="/MT/translate/hakka_zh_hk",
                    text=text[:300],
                    token=token,
                )
                full_hakka = result.get("output", "").strip()
                if not full_hakka:
                    return "", ""
                # TTS 限制 200 字
                audio = await generate_hakka_tts(full_hakka[:200], folder="ocr_full")
                return full_hakka, audio
            except Exception as e:
                print(f"[OCR] 全文翻譯失敗：{e}")
                return "", ""

        async def _translate_one(word: str) -> KeywordItem:
            """單一詞彙：中文 → 客語漢字 → 客語拼音 + TTS，三步串聯"""
            try:
                token = await get_trans_token()

                # Step 1：中文 → 客語漢字
                result = await call_hakka_translate_api(
                    endpoint="/MT/translate/hakka_zh_hk",
                    text=word,
                    token=token,
                )
                hakka = result.get("output", "").strip() or word

                # Step 2：客語漢字 → 客語拼音（並行執行 TTS）
                async def _get_pinyin() -> str:
                    try:
                        py_result = await call_hakka_translate_api(
                            endpoint="/MT/translate/hakka_hk_py",
                            text=hakka,
                            token=token,
                        )
                        return py_result.get("output", "").strip()
                    except Exception as e:
                        print(f"[OCR] 詞彙「{word}」拼音失敗：{e}")
                        return ""

                async def _get_tts() -> str:
                    try:
                        return await generate_hakka_tts(hakka, folder="ocr_keywords")
                    except Exception as e:
                        print(f"[OCR] 詞彙「{word}」TTS 失敗：{e}")
                        return ""

                # 拼音和 TTS 並行，不互相等待
                pinyin, audio = await asyncio.gather(_get_pinyin(), _get_tts())

                return KeywordItem(
                    word_zh=word,
                    word_hakka=hakka,
                    word_pinyin=pinyin,
                    audio_path=audio,
                )
            except Exception as e:
                print(f"[OCR] 詞彙「{word}」翻譯失敗：{e}")
                return KeywordItem(word_zh=word, word_hakka="", word_pinyin="", audio_path="")

        # ── 3. LLM 抽關鍵詞（與全文翻譯並行）──────────────────────────────
        keywords_zh = await _extract_keywords_llm(text)
        print(f"[OCR] 抽出關鍵詞：{keywords_zh}")

        # 全文翻譯 + 所有詞彙翻譯 全部並行執行
        tasks = [_full_translate()] + [_translate_one(w) for w in keywords_zh]
        results = await asyncio.gather(*tasks)

        full_hakka, full_audio_path = results[0]
        keyword_items = list(results[1:])

        return OcrWithKeywordsResult(
            text=text,
            engine=engine,
            char_count=len(text.replace(" ", "")),
            full_hakka=full_hakka,
            full_audio_path=full_audio_path,
            keywords=keyword_items,
        )

    finally:
        if _os.path.exists(tmp_path):
            _os.unlink(tmp_path)
