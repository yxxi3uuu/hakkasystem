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


def init_llm() -> None:
    """由 main.py lifespan 在 .env 確定載入後呼叫，初始化全域 llm 物件。"""
    global llm

    _model_path = os.getenv("LLM_MODEL_PATH", "").strip()

    if not _model_path:
        print("[LLM] LLM_MODEL_PATH 未設定，將使用 fallback 例句庫")
        return

    if not os.path.exists(_model_path):
        print(f"[LLM] 模型路徑不存在：{_model_path}（將使用 fallback）")
        return

    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=_model_path,
            n_gpu_layers=20,
            n_ctx=2048,
            n_batch=256,
            flash_attn=True,
            verbose=False,
        )
        print(f"[LLM] 模型載入成功：{_model_path}")
    except Exception as e:
        print(f"[LLM] 載入失敗（將使用 fallback）: {e}")


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
    return json.loads(_clean_json_text(result_text))


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
        system_prompt = """你是一個專業的台灣客語教師。
請根據輸入的中文單字，生成一句適合國小生學習的生活化客語例句，長度在 10 個字以內。
請務必只輸出合法 JSON，包含 "hakka_sentence" 與 "chinese_translation" 兩個鍵值，絕對不要輸出任何其他說明文字。"""
        user_prompt = f"""範例：
輸入：椅子
{{"hakka_sentence": "這張椅子當好坐。", "chinese_translation": "這張椅子很好坐。"}}

輸入：吃飯
{{"hakka_sentence": "大家來食飯囉。", "chinese_translation": "大家來吃飯囉。"}}

輸入：{word}"""
        try:
            return _ask_llm_json(system_prompt, user_prompt, max_tokens=150, temperature=0.2)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="模型未輸出正確的 JSON 格式")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推論錯誤: {e}")

    # fallback
    return FALLBACK.get(word, {
        "hakka_sentence":      f"這個{word}當靚。",
        "chinese_translation": f"這個{word}很漂亮。",
    })


# ── 生成多單字故事 ────────────────────────────────────────────────────────
@router.post("/generate-story")
async def generate_hakka_story(request: WordsRequest):
    words = _normalize_words(request.words)
    if not words:
        raise HTTPException(status_code=400, detail="單字列表不可為空")

    words_str = "、".join(words)

    if llm:
        system_prompt = """你是一個專業的中文寫作助手。請根據我提供的 1 到 5 個中文單字，為每個單字造一個中文句子。
要求：
1. 這些句子必須構成一個有關聯的連續情境或故事。
2. 每個句子的長度必須控制在大約 20 個字左右。
3. 請務必只輸出合法的 JSON 陣列 (Array) 格式，包含 "word" 與 "sentence" 兩個鍵值，絕對不要輸出任何其他說明文字。"""
        user_prompt = f"""範例：
輸入單字：蘋果、公園、下雨
[
  {{"word": "蘋果", "sentence": "他手裡拿著一顆紅透的蘋果，心情看起來非常愉快。"}},
  {{"word": "公園", "sentence": "我們原本約好要在這座寬敞的公園裡一起野餐吃水果。"}},
  {{"word": "下雨", "sentence": "沒想到天空突然下雨，打亂了所有原本規劃好的行程。"}}
]

輸入單字：{words_str}"""
        try:
            parsed = _ask_llm_json(system_prompt, user_prompt, max_tokens=400, temperature=0.3)
            if not isinstance(parsed, list):
                raise json.JSONDecodeError("Expected JSON array", str(parsed), 0)
            return _with_compatible_sentence_fields(parsed, words)
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
