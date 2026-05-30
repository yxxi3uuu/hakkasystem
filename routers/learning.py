"""
routers/learning.py
客語例句 / 故事生成
使用本地 llama_cpp 模型，路徑從環境變數 LLM_MODEL_PATH 讀取。
若模型不存在或未設定，自動 fallback 到內建範例句庫。
"""

import json
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning")


class WordRequest(BaseModel):
    word: str


class WordsRequest(BaseModel):
    words: list[str]


# ── 載入本地 LLM（路徑從環境變數讀取）──────────────────────────────────────
llm = None

_model_path = os.getenv("LLM_MODEL_PATH", "").strip()

if _model_path:
    if not os.path.exists(_model_path):
        print(f"[LLM] 模型路徑不存在：{_model_path}（將使用 fallback）")
    else:
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
else:
    print("[LLM] LLM_MODEL_PATH 未設定，將使用 fallback 例句庫")


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
        prompt = f"""[INST] 你是一個專業的台灣客語教師。請根據輸入的中文單字，生成一句適合國小生學習的生活化客語例句（長度在 10 個字以內）。
請務必只輸出合法的 JSON 格式，包含 "hakka_sentence" 與 "chinese_translation" 兩個鍵值，絕對不要輸出任何其他說明文字。

輸入：椅子 [/INST]
{{"hakka_sentence": "這張椅子當好坐。", "chinese_translation": "這張椅子很好坐。"}}

[INST] 輸入：吃飯 [/INST]
{{"hakka_sentence": "大家來食飯囉。", "chinese_translation": "大家來吃飯囉。"}}

[INST] 輸入：{word} [/INST]
"""
        try:
            backticks = chr(96) * 3
            response = llm(
                prompt,
                max_tokens=150,
                temperature=0.2,
                echo=False,
                stop=["[INST]", backticks],
            )
            result_text = response["choices"][0]["text"].strip()
            cleaned = (result_text
                       .replace(f"{backticks}json\n", "")
                       .replace(f"{backticks}json", "")
                       .replace(backticks, "")
                       .strip())
            return json.loads(cleaned)
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
    words = [w.strip() for w in request.words if w.strip()]
    if not words:
        raise HTTPException(status_code=400, detail="單字列表不可為空")

    words_str = "、".join(words)

    if llm:
        prompt = f"""[INST] 你是一個專業的「客語故事寫作助手」。請根據我提供的多個單字，為每個單字造一個客語句子。
要求：
1. 【最重要】這些句子必須構成一個「有前後因果關係、流暢的連續情境或故事」！絕對不要各說各話的獨立造句。
2. 每個句子的長度必須控制在 10 到 20 個字左右，適合國小生學習。
3. 每個單字對應的輸出，都必須包含客語發音句 (hakka_sentence) 以及白話文翻譯 (chinese_translation)。
4. 請務必只輸出合法的 JSON 陣列 (Array) 格式，包含 "word", "hakka_sentence" 與 "chinese_translation" 三個鍵值，絕對不要輸出任何其他說明文字。

輸入單字：蘋果、公園、下雨 [/INST]
[
  {{"word": "蘋果", "hakka_sentence": "他手項拿一粒蘋果，當歡喜。", "chinese_translation": "他手裡拿著一顆紅透的蘋果，看起來非常愉快。"}},
  {{"word": "公園", "hakka_sentence": "𠊎兜原本約好要在這大公園食水菓。", "chinese_translation": "我們原本約好要在這座寬敞的公園裡一起野餐吃水果。"}},
  {{"word": "下雨", "hakka_sentence": "沒想到天公突然落雨，打亂了行程。", "chinese_translation": "沒想到天空突然下雨，打亂了所有原本規劃好的行程。"}}
]

[INST] 輸入單字：{words_str} [/INST]
"""
        try:
            backticks = chr(96) * 3
            response = llm(
                prompt,
                max_tokens=400,
                temperature=0.3,
                echo=False,
                stop=["[INST]", backticks],
            )
            result_text = response["choices"][0]["text"].strip()
            cleaned = (result_text
                       .replace(f"{backticks}json\n", "")
                       .replace(f"{backticks}json", "")
                       .replace(backticks, "")
                       .strip())
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="模型未輸出正確的 JSON 格式")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推論錯誤: {e}")

    # fallback
    return [
        {
            "word": w,
            **FALLBACK.get(w, {
                "hakka_sentence":      f"這個{w}當靚。",
                "chinese_translation": f"這個{w}很漂亮。",
            }),
        }
        for w in words
    ]
