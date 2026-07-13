"""
routers/learning.py
客語例句 / 故事生成
使用本地 llama_cpp 模型，路徑從環境變數 LLM_MODEL_PATH 讀取。
若模型不存在或未設定，自動 fallback 到內建範例句庫。
"""

import json
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning")


class WordRequest(BaseModel):
    word: str


class WordsRequest(BaseModel):
    words: list[str] | str


# ── 載入本地 LLM（由 main.py lifespan 呼叫 init_llm() 完成初始化）────────
llm = None


def _candidate_model_paths(raw_path: str) -> list[Path]:
    """Return possible model paths, ordered from most explicit to most portable."""
    candidates: list[Path] = []
    path = Path(raw_path).expanduser()

    candidates.append(path)

    if not path.is_absolute():
        candidates.append(Path(__file__).resolve().parent.parent / path)

    candidates.append(Path(__file__).resolve().parent.parent / "data" / path.name)
    return candidates


def init_llm() -> None:
    """由 main.py lifespan 在 .env 確定載入後呼叫，初始化全域 llm 物件。"""
    global llm

    _model_path = os.getenv("LLM_MODEL_PATH", "").strip()

    if not _model_path:
        print("[LLM] LLM_MODEL_PATH 未設定，將使用 fallback 例句庫")
        return

    candidates = _candidate_model_paths(_model_path)
    print(f"[LLM] 嘗試模型路徑：{_model_path}，候選路徑：{[str(p) for p in candidates]}")

    resolved_path = None
    for candidate in candidates:
        if candidate.exists():
            resolved_path = candidate
            break

    if resolved_path is None:
        print(f"[LLM] 找不到模型檔：{_model_path}（候選路徑：{[str(p) for p in candidates]}，將使用 fallback）")
        return

    try:
        from llama_cpp import Llama
    except ImportError as e:
        print(f"[LLM] llama_cpp 未安裝，無法載入模型：{e}（將使用 fallback）")
        return

    try:
        llm = Llama(
            model_path=str(resolved_path),
            n_gpu_layers=20,
            n_ctx=2048,
            n_batch=256,
            flash_attn=True,
            verbose=False,
        )
        print(f"[LLM] 模型載入成功：{resolved_path}")
    except Exception as e:
        print(f"[LLM] 載入失敗（將使用 fallback）: {resolved_path} / {e}")


def _clean_json_text(text: str) -> str:
    backticks = chr(96) * 3
    cleaned = (
        text.replace(f"{backticks}json\n", "")
        .replace(f"{backticks}json", "")
        .replace(backticks, "")
        .strip()
    )
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
    return match.group(1).strip() if match else cleaned


def _coerce_sentence_result(parsed: object, word: str) -> dict:
    if isinstance(parsed, dict):
        sentence = (
            parsed.get("sentence_zh")
            or parsed.get("sentence")
            or parsed.get("chinese_translation")
            or ""
        )
        if isinstance(sentence, str) and sentence.strip():
            return {
                "word": parsed.get("word") or word,
                "sentence_zh": sentence.strip(),
                "sentence": sentence.strip(),
                "chinese_translation": sentence.strip(),
            }

    fallback_sentence = f"這是一個關於{word}的生活句子。"
    return {
        "word": word,
        "sentence_zh": fallback_sentence,
        "sentence": fallback_sentence,
        "chinese_translation": fallback_sentence,
    }


def _ask_llm_json(system_prompt: str, user_prompt: str, max_tokens: int, temperature: float):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result_text = response["choices"][0]["message"]["content"].strip()
    cleaned_text = _clean_json_text(result_text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {"sentence_zh": cleaned_text or result_text, "sentence": cleaned_text or result_text, "word": ""}


def _normalize_words(words: list[str] | str, max_words: int = 5) -> list[str]:
    if isinstance(words, str):
        raw_words = words.split()
    else:
        raw_words = []
        for item in words:
            raw_words.extend(str(item).split())

    return [word.strip() for word in raw_words if word.strip()][:max_words]


def _with_compatible_sentence_fields(items: list[dict], words: list[str]) -> list[dict]:
    results = []

    for index, word in enumerate(words):
        item = items[index] if index < len(items) and isinstance(items[index], dict) else {}
        sentence = (
            item.get("sentence")
            or item.get("sentence_zh")
            or item.get("chinese_translation")
            or f"這是一個{word}。"
        )

        results.append({
            "word": item.get("word") or word,
            "sentence": sentence,
            "sentence_zh": sentence,
            "chinese_translation": sentence,
        })

    return results


# ── Fallback 範例句庫 ─────────────────────────────────────────────────────
FALLBACK: dict[str, dict] = {
    "人":   {"hakka_sentence": "這個人當好。",       "chinese_translation": "這個人很好。"},
    "貓":   {"hakka_sentence": "隻貓仔當可愛。",     "chinese_translation": "這隻貓很可愛。"},
    "狗":   {"hakka_sentence": "隻狗仔當乖。",       "chinese_translation": "這隻狗很乖。"},
    "椅子": {"hakka_sentence": "這張椅子當好坐。",   "chinese_translation": "這張椅子很好坐。"},
    "杯子": {"hakka_sentence": "這個杯仔當靚。",     "chinese_translation": "這個杯子很漂亮。"},
    "書":   {"hakka_sentence": "這本書當好睇。",     "chinese_translation": "這本書很好看。"},
    "手機": {"hakka_sentence": "這支手機當新。",     "chinese_translation": "這支手機很新。"},
    "蘋果": {"hakka_sentence": "這粒蘋果當甜。",     "chinese_translation": "這顆蘋果很甜。"},
}


# ── 生成單字例句 ─────────────────────────────────────────────────────────
@router.post("/generate-sentence")
async def generate_hakka_sentence(request: WordRequest):
    word = request.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="單字不可為空")

    if llm:
        system_prompt = """你是一個專業的中文句子生成助手。
請根據輸入的中文單字，生成一句生活化、適合國小生的中文句子。
要求：
1. 只輸出一個中文句子，不要任何英文或客語。
2. 長度約 10 到 20 個中文字。
3. 內容要自然、生活化、容易理解。
4. 不要輸入任何額外說明文字。"""
        user_prompt = f"""範例：
輸入：椅子
這張椅子坐起來很舒服。

輸入：吃飯
大家一起吃飯，氣氛很溫暖。

輸入：{word}"""
        try:
            parsed = _ask_llm_json(system_prompt, user_prompt, max_tokens=120, temperature=0.2)
            if isinstance(parsed, dict):
                return _coerce_sentence_result(parsed, word)
            return _coerce_sentence_result({}, word)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="模型未輸出正確的 JSON 格式")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推論錯誤: {e}")

    # fallback
    fallback_sentence = f"這是一個關於{word}的生活句子。"
    return {
        "word": word,
        "sentence_zh": fallback_sentence,
        "sentence": fallback_sentence,
        "chinese_translation": fallback_sentence,
    }


# ── 生成多單字故事 ────────────────────────────────────────────────────────
@router.post("/generate-story")
async def generate_hakka_story(request: WordsRequest):
    words = _normalize_words(request.words)
    if not words:
        raise HTTPException(status_code=400, detail="單字列表不可為空")

    words_str = "、".join(words)

    if llm:
        system_prompt = """你是一個專業的中文句子生成助手。請根據我提供的 1 到 5 個中文單字，為每個單字各造一個中文生活句子。
要求：
1. 每個句子都要是中文，長度約 10 到 20 個中文字。
2. 內容要自然、生活化、適合國小生。
3. 只輸出合法的 JSON 陣列，陣列中每一項只包含兩個鍵：\"word\" 與 \"sentence_zh\"。
4. 不要輸入任何額外說明文字。"""
        user_prompt = f"""範例：
輸入單字：蘋果、公園、下雨
[
  {{"word": "蘋果", "sentence_zh": "這顆蘋果看起來又紅又香。"}},
  {{"word": "公園", "sentence_zh": "我們在公園裡一起散步。"}},
  {{"word": "下雨", "sentence_zh": "突然下雨了，大家趕快回家。"}}
]

輸入單字：{words_str}"""
        try:
            parsed = _ask_llm_json(system_prompt, user_prompt, max_tokens=400, temperature=0.2)
            if not isinstance(parsed, list):
                return [
                    _coerce_sentence_result({}, w) for w in words
                ]
            results = []
            for idx, word in enumerate(words):
                item = parsed[idx] if idx < len(parsed) and isinstance(parsed[idx], dict) else {}
                results.append(_coerce_sentence_result(item, word))
            return results
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="模型未輸出正確的 JSON 格式")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推論錯誤: {e}")

    # fallback
    return [
        {
            "word": w,
            "sentence": f"這是一個關於{w}的生活句子。",
            "sentence_zh": f"這是一個關於{w}的生活句子。",
            "chinese_translation": f"這是一個關於{w}的生活句子。",
        }
        for w in words
    ]
