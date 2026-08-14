"""
routers/learning.py
客語例句 / 故事生成
優先使用 Gemini API 造句，若 Gemini 不可用則 fallback 到罐頭句。
"""

import json
import logging
import os
import re
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning")


class WordRequest(BaseModel):
    word: str


class WordsRequest(BaseModel):
    words: list[str] | str


# ── LLM 造句設定（優先 Groq，備選 Gemini）─────────────────────────────────
_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
_GEMINI_MODEL = "gemini-2.0-flash"
_LLM_BACKOFF_UNTIL = 0.0


def init_llm() -> None:
    """保留介面相容性，由 main.py lifespan 呼叫。"""
    global _GROQ_API_KEY, _GEMINI_API_KEY
    _GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    _GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

    if _GROQ_API_KEY:
        print(f"[LLM] 使用 Groq API 造句（key={_GROQ_API_KEY[:8]}...）")
    elif _GEMINI_API_KEY:
        print(f"[LLM] 使用 Gemini API 造句（model={_GEMINI_MODEL}）")
    else:
        print("[LLM] 未設定任何 LLM API Key，造句將使用 fallback 罐頭句")


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


def _coerce_sentence_result(parsed: object, word: str, index: int = 0) -> dict:
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

    fallback_sentence = _fallback_sentence(word, index)
    return {
        "word": word,
        "sentence_zh": fallback_sentence,
        "sentence": fallback_sentence,
        "chinese_translation": fallback_sentence,
    }


def _normalize_words(words: list[str] | str, max_words: int = 5) -> list[str]:
    if isinstance(words, str):
        raw_words = words.split()
    else:
        raw_words = []
        for item in words:
            raw_words.extend(str(item).split())

    return [word.strip() for word in raw_words if word.strip()][:max_words]


async def _ask_llm(prompt: str, max_tokens: int = 300) -> str:
    """呼叫 LLM API 造句。優先 Groq，備選 Gemini，都失敗回傳空字串。"""
    import asyncio
    global _LLM_BACKOFF_UNTIL

    if time.monotonic() < _LLM_BACKOFF_UNTIL:
        return ""

    # ── 優先用 Groq（免費、穩定）──
    if _GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {_GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "openai/gpt-oss-20b",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM] Groq 造句失敗：{e}")
            # Groq 失敗 fallthrough 到 Gemini

    # ── 備選 Gemini ──
    if _GEMINI_API_KEY:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": max_tokens,
            },
        }

        last_err = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(url, params={"key": _GEMINI_API_KEY}, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                candidates = data.get("candidates", [])
                if not candidates:
                    return ""
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return ""
                return parts[0].get("text", "").strip()
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    _LLM_BACKOFF_UNTIL = time.monotonic() + 60
                    print("[LLM] Gemini 已達請求限制，60 秒內改用本地備用例句")
                    return ""
                if any(code in err_str for code in ("503", "UNAVAILABLE")):
                    if attempt < 2:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                print(f"[LLM] Gemini 造句失敗：{e}")
                return ""

        print(f"[LLM] Gemini 造句重試 3 次仍失敗：{last_err}")

    return ""


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

GENERIC_FALLBACK_TEMPLATES = (
    "我今天在生活中看見了{word}。",
    "請你找找看，{word}在什麼地方？",
    "我們一起來認識{word}的客語講法。",
    "照片裡的{word}是今天要學的詞。",
    "你在日常生活中有看過{word}嗎？",
)


def _fallback_sentence(word: str, index: int = 0) -> str:
    """Return a useful local sentence without presenting AI output as fact."""
    known = FALLBACK.get(word)
    if known and known.get("chinese_translation"):
        return str(known["chinese_translation"]).strip()
    template = GENERIC_FALLBACK_TEMPLATES[index % len(GENERIC_FALLBACK_TEMPLATES)]
    return template.format(word=word)


# ── 生成單字例句 ─────────────────────────────────────────────────────────
@router.post("/generate-sentence")
async def generate_hakka_sentence(request: WordRequest):
    word = request.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="單字不可為空")

    if _GROQ_API_KEY or _GEMINI_API_KEY:
        prompt = f"""你是一個專業的中文句子生成助手。
請根據輸入的中文單字，生成一句生活化、適合國小生的中文句子。
要求：
1. 只輸出一個中文句子，不要任何英文或客語。
2. 長度約 10 到 20 個中文字。
3. 內容要自然、生活化、容易理解。
4. 不要輸出任何額外說明文字，只輸出句子本身。

範例：
輸入：椅子
這張椅子坐起來很舒服。

輸入：吃飯
大家一起吃飯，氣氛很溫暖。

輸入：{word}"""
        try:
            result = await _ask_llm(prompt, max_tokens=100)
            if result:
                # Gemini 直接回傳句子，不是 JSON
                sentence = result.strip().split("\n")[0].strip()
                return {
                    "word": word,
                    "sentence_zh": sentence,
                    "sentence": sentence,
                    "chinese_translation": sentence,
                }
        except Exception as e:
            print(f"[LLM] Gemini 造句失敗：{e}")

    # fallback
    fallback_sentence = _fallback_sentence(word)
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

    if _GROQ_API_KEY or _GEMINI_API_KEY:
        prompt = f"""你是一個專業的中文句子生成助手。請根據我提供的中文單字，為每個單字各造一個中文生活句子。
要求：
1. 每個句子都要是中文，長度約 10 到 20 個中文字。
2. 內容要自然、生活化、適合國小生。
3. 只輸出合法的 JSON 陣列，不要有任何額外說明。
4. 格式如下：
[
  {{"word": "蘋果", "sentence_zh": "這顆蘋果看起來又紅又香。"}},
  {{"word": "公園", "sentence_zh": "我們在公園裡一起散步。"}}
]

輸入單字：{words_str}"""
        try:
            result = await _ask_llm(prompt, max_tokens=400)
            if result:
                cleaned = _clean_json_text(result)
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    results = []
                    for idx, word in enumerate(words):
                        item = parsed[idx] if idx < len(parsed) and isinstance(parsed[idx], dict) else {}
                        results.append(_coerce_sentence_result(item, word, idx))
                    return results
        except Exception as e:
            print(f"[LLM] Gemini 批次造句失敗：{e}")

    # fallback
    results = []
    for index, word in enumerate(words):
        sentence = _fallback_sentence(word, index)
        results.append(
            {
                "word": word,
                "sentence": sentence,
                "sentence_zh": sentence,
                "chinese_translation": sentence,
                "sentence_source": "local_fallback",
            }
        )
    return results
