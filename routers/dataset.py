"""
routers/dataset.py
客語邊角案例資料收集 API
第一道防線：收到資料後，背景呼叫 LLM 對 correct_text 進行自動校正預審。
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dataset", tags=["客語資料集收集"])

# ── 儲存路徑 ──────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("dataset_storage")
AUDIO_DIR     = DATASET_DIR / "audios"
METADATA_FILE = DATASET_DIR / "metadata.jsonl"

# ── OpenAI 相容設定（支援 OpenAI / Azure / 任何相容介面）────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# AI 校正 Prompt 模板
_AI_PROMPT_TEMPLATE = (
    "你是一位精通台灣客語（含四縣、海陸等腔調）的語言學專家。"
    "使用者嘗試拼寫一個客語詞彙：'{correct_text}'，"
    "這原本是要用來修正 '{wrong_text}' 的。"
    "請判斷使用者的客語漢字或拼寫是否符合教育部台灣客語辭典的標準規範？"
    "如果不符合，請直接幫他修正，並「只回傳修正後的客語漢字/詞彙」；"
    "如果完全正確，請原樣回傳。"
    "切記：不要有任何額外的解釋、引號或標點符號，只需要回傳最終的客語字。"
)


def _ensure_dirs() -> None:
    """確保儲存目錄存在。"""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


async def _ai_review(wrong_text: str, correct_text: str) -> str:
    """
    呼叫 LLM 對 correct_text 進行客語標準化校正。
    若 API Key 未設定或呼叫失敗，靜默 fallback 回傳原始 correct_text，
    不阻斷主流程。
    """
    if not OPENAI_API_KEY:
        logger.warning("[AI預審] OPENAI_API_KEY 未設定，跳過 AI 校正，直接使用原始輸入。")
        return correct_text

    prompt = _AI_PROMPT_TEMPLATE.format(
        correct_text=correct_text,
        wrong_text=wrong_text,
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,       # 校正任務要確定性輸出
        "max_tokens": 64,       # 客語詞彙不會太長
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            ai_text = data["choices"][0]["message"]["content"].strip()
            logger.info("[AI預審] wrong='%s' | user='%s' → ai='%s'",
                        wrong_text, correct_text, ai_text)
            return ai_text

    except httpx.HTTPStatusError as e:
        logger.error("[AI預審] HTTP 錯誤 %s：%s", e.response.status_code, e.response.text)
    except Exception as e:
        logger.error("[AI預審] 呼叫失敗：%s", e)

    # 任何例外都 fallback，不影響資料收集
    return correct_text


@router.post("/collect", summary="收集客語邊角案例資料（含 AI 預審）")
async def collect_dataset(
    audio: UploadFile = File(..., description="使用者的客語錄音檔"),
    wrong_text: str   = Form(..., description="當時 API 誤判的文字"),
    correct_text: str = Form(..., description="使用者或手動修正的正確客語字"),
    scenario: str     = Form(..., description="誤判情境標記，例如 complex_background / dialect_missing"),
):
    """
    收集一筆邊角案例：
    1. 儲存音訊至 dataset_storage/audios/<uuid>.<ext>
    2. 呼叫 LLM 對 correct_text 進行客語標準化預審
    3. 將所有欄位（含 ai_suggested_text、review_status=pending）
       append 至 dataset_storage/metadata.jsonl
    """
    _ensure_dirs()

    # ── 1. 儲存音訊檔 ─────────────────────────────────────────────────────────
    original_filename = audio.filename or "audio"
    suffix = Path(original_filename).suffix or ".wav"
    audio_filename = f"{uuid.uuid4().hex}{suffix}"
    audio_path = AUDIO_DIR / audio_filename

    try:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="上傳的音訊檔案為空")
        audio_path.write_bytes(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音訊儲存失敗：{e}") from e

    # ── 2. AI 預審（第一道防線）──────────────────────────────────────────────
    ai_suggested_text = await _ai_review(wrong_text, correct_text)

    # ── 3. 組裝標註資料 ───────────────────────────────────────────────────────
    record = {
        "id":                uuid.uuid4().hex,
        "audio_path":        str(audio_path).replace("\\", "/"),
        "wrong_text":        wrong_text,
        "correct_text":      correct_text,          # 使用者原始填寫
        "ai_suggested_text": ai_suggested_text,     # AI 校正建議
        "review_status":     "pending",             # 等待管理員人工終審
        "scenario":          scenario,
        "original_filename": original_filename,
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }

    # ── 4. Append 至 metadata.jsonl ──────────────────────────────────────────
    try:
        with METADATA_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata 寫入失敗：{e}") from e

    return {
        "success": True,
        "message": "資料收集成功，感謝您的貢獻！",
        "data": {
            "id":                record["id"],
            "audio_path":        record["audio_path"],
            "scenario":          scenario,
            "ai_suggested_text": ai_suggested_text,
            "review_status":     "pending",
        },
    }
