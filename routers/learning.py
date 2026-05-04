from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json

router = APIRouter(prefix="/api/learning")

class WordRequest(BaseModel):
    word: str

# 嘗試載入 LLM，失敗時用 fallback
llm = None
try:
    from llama_cpp import Llama
    llm = Llama(
        model_path=r"C:\Users\user\Desktop\新增資料夾 (3)\Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
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
            response = llm(prompt, max_tokens=100, temperature=0.2, echo=False)
            result_text = response["choices"][0]["text"].strip()
            return json.loads(result_text)
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
