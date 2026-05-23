from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json

router = APIRouter(prefix="/api/learning")

class WordRequest(BaseModel):
    word: str

class WordsRequest(BaseModel):
    words: list[str]

# 嘗試載入 LLM，失敗時用 fallback
llm = None
try:
    from llama_cpp import Llama
    llm = Llama(
        model_path=r"C:\Users\user\Desktop\llm test\Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        n_gpu_layers=20,
        n_ctx=2048,
        n_batch=256,
        flash_attn=True,
        verbose=False,
    )
except Exception as e:
    print(f"LLM 載入失敗（將使用 fallback）: {e}")

# Fallback 範例句庫
FALLBACK = {
    "人":   {"hakka_sentence": "這個人當好。",     "chinese_translation": "這個人很好。"},
    "貓":   {"hakka_sentence": "隻貓仔當可愛。",   "chinese_translation": "這隻貓很可愛。"},
    "狗":   {"hakka_sentence": "隻狗仔當乖。",     "chinese_translation": "這隻狗很乖。"},
    "椅子": {"hakka_sentence": "這張椅子當好坐。", "chinese_translation": "這張椅子很好坐。"},
    "杯子": {"hakka_sentence": "這個杯仔當靚。",   "chinese_translation": "這個杯子很漂亮。"},
    "書":   {"hakka_sentence": "這本書當好睇。",   "chinese_translation": "這本書很好看。"},
    "手機": {"hakka_sentence": "這支手機當新。",   "chinese_translation": "這支手機很新。"},
    "蘋果": {"hakka_sentence": "這粒蘋果當甜。",   "chinese_translation": "這顆蘋果很甜。"},
}

@router.post("/generate-sentence")
async def generate_hakka_sentence(request: WordRequest):
    word = request.word

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
            response = llm(
                prompt, 
                max_tokens=150, 
                temperature=0.2, 
                echo=False,
                stop=["[INST]", "```"]
            )
            backticks = chr(96) * 3
            result_text = response["choices"][0]["text"].strip()
            cleaned_text = result_text.replace(f"{backticks}json\n", "").replace(f"{backticks}json", "").replace(backticks, "").strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="模型未輸出正確的 JSON 格式")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推論錯誤: {str(e)}")

    # LLM 未載入時用 fallback
    fallback = FALLBACK.get(word, {
        "hakka_sentence": f"這個{word}當靚。",
        "chinese_translation": f"這個{word}很漂亮。",
    })
    return fallback

@router.post("/generate-story")
async def generate_hakka_story(request: WordsRequest):
    words = request.words
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
            response = llm(
                prompt,
                max_tokens=400,
                temperature=0.3,
                echo=False,
                stop=["[INST]", "```"]
            )
            backticks = chr(96) * 3
            result_text = response["choices"][0]["text"].strip()
            cleaned_text = result_text.replace(f"{backticks}json\n", "").replace(f"{backticks}json", "").replace(backticks, "").strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="模型未輸出正確的 JSON 格式")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推論錯誤: {str(e)}")

    # LLM 未載入時用 fallback
    fallback_list = []
    for w in words:
        fallback_item = FALLBACK.get(w, {
            "hakka_sentence": f"這個{w}當靚。",
            "chinese_translation": f"這個{w}很漂亮。"
        })
        fallback_list.append({"word": w, **fallback_item})
    return fallback_list
