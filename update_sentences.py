"""
update_sentences.py
把資料庫中 sentence_zh 是 fallback 格式的記錄，重新用 LLM + 客語 API 更新。

使用方式：
    conda activate hakka311
    python update_sentences.py

選項：
    --dry-run   只列出要更新的資料，不實際更新
    --user-id N 只更新特定使用者的資料
"""

import asyncio
import argparse
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# 載入 .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import asyncpg


# ── 判斷是否為 fallback 句子 ──────────────────────────────────────────
def is_fallback_sentence(sentence: str) -> bool:
    if not sentence:
        return True
    patterns = [
        r"^這是一個關於.+的生活句子[。.]?$",
        r"^This is a sentence about",
    ]
    for p in patterns:
        if re.match(p, sentence.strip()):
            return True
    return False


# ── 呼叫客語翻譯 API ─────────────────────────────────────────────────
async def get_hakka_translation(text: str, endpoint: str, token: str) -> str:
    import httpx
    hktrans = os.getenv("HKTRANS_API", "https://hktrans.bronci.com.tw")
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{hktrans}{endpoint}",
            json={"input": text},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
    data = res.json()
    return data.get("output", "").strip()


async def get_trans_token() -> str:
    import httpx
    hktrans   = os.getenv("HKTRANS_API", "https://hktrans.bronci.com.tw")
    username  = os.getenv("HAKKA_USERNAME")
    password  = os.getenv("HAKKA_PASSWORD")
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{hktrans}/api/v1/tts/login",
            json={"username": username, "password": password, "rememberMe": 1},
            timeout=10.0,
        )
    data = res.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"客語 API 登入失敗：{data}")
    return token


# ── 呼叫 LLM 造句 ─────────────────────────────────────────────────────
def ask_llm_sentence(llm, word: str) -> str:
    system_prompt = """你是一個專業的中文寫作助手。請根據我提供的中文單字，造一個生活化、有情境的中文句子。
要求：
1. 句子長度控制在 20 個字左右。
2. 請務必只輸出合法 JSON，包含 "sentence" 鍵值，不要輸出任何說明文字。"""
    user_prompt = f"""範例：
輸入：蘋果
{{"sentence": "他手裡拿著一顆紅透的蘋果，心情看起來非常愉快。"}}

輸入：{word}"""
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=120,
        temperature=0.3,
    )
    text = response["choices"][0]["message"]["content"].strip()
    # 清理 JSON
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        import json
        data = json.loads(match.group())
        return data.get("sentence", "").strip()
    return ""


# ── 主程式 ────────────────────────────────────────────────────────────
async def main(dry_run: bool, user_id_filter: int | None):
    db_url = os.getenv("DATABASE_URL", "")
    # asyncpg 不支援 SQLAlchemy 格式，轉換
    pg_url = db_url.replace("postgresql+asyncpg://", "postgresql://").split("?")[0]

    print(f"連接資料庫：{pg_url}")
    conn = await asyncpg.connect(pg_url)

    try:
        # 查找 fallback 句子
        if user_id_filter:
            rows = await conn.fetch(
                "SELECT id, label_zh, sentence_zh, sentence_hakka FROM saved_words WHERE user_id=$1",
                user_id_filter
            )
        else:
            rows = await conn.fetch(
                "SELECT id, label_zh, sentence_zh, sentence_hakka FROM saved_words"
            )

        targets = [r for r in rows if is_fallback_sentence(r["sentence_zh"])]
        print(f"找到 {len(targets)} 筆需要更新的記錄（共 {len(rows)} 筆）")

        if not targets:
            print("沒有需要更新的資料")
            return

        if dry_run:
            print("\n[Dry Run] 以下記錄將被更新：")
            for r in targets:
                print(f"  ID={r['id']} label_zh={r['label_zh']} | 舊句子：{r['sentence_zh']}")
            return

        # 初始化 LLM
        llm_path = os.getenv("LLM_MODEL_PATH", "").strip()
        if not llm_path:
            print("錯誤：LLM_MODEL_PATH 未設定")
            sys.exit(1)

        # 解析路徑
        model_file = Path(llm_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent / model_file
        if not model_file.exists():
            model_file = Path(__file__).resolve().parent / "data" / Path(llm_path).name
        if not model_file.exists():
            print(f"錯誤：找不到模型檔：{llm_path}")
            sys.exit(1)

        print(f"載入 LLM：{model_file}")
        from llama_cpp import Llama
        llm = Llama(
            model_path=str(model_file),
            n_gpu_layers=20,
            n_ctx=2048,
            n_batch=256,
            verbose=False,
        )
        print("LLM 載入成功")

        # 取得客語 API token
        print("取得客語 API token...")
        token = await get_trans_token()
        print("Token 取得成功")

        # 逐筆更新
        updated = 0
        failed  = 0
        for r in targets:
            word = r["label_zh"]
            word_id = r["id"]
            print(f"\n處理 ID={word_id} [{word}]...")

            try:
                # 1. LLM 造句
                sentence_zh = ask_llm_sentence(llm, word)
                if not sentence_zh:
                    print(f"  ⚠️  LLM 造句失敗，跳過")
                    failed += 1
                    continue
                print(f"  中文句子：{sentence_zh}")

                # 2. 翻譯成客語
                sentence_hakka = await get_hakka_translation(
                    sentence_zh, "/MT/translate/hakka_zh_hk", token
                )
                print(f"  客語句子：{sentence_hakka}")

                # 3. 更新資料庫
                await conn.execute(
                    "UPDATE saved_words SET sentence_zh=$1, sentence_hakka=$2 WHERE id=$3",
                    sentence_zh, sentence_hakka, word_id
                )
                updated += 1
                print(f"  ✅ 已更新")

            except Exception as e:
                print(f"  ❌ 失敗：{e}")
                failed += 1

        print(f"\n完成！更新 {updated} 筆，失敗 {failed} 筆")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="更新 fallback 句子為 LLM 生成版本")
    parser.add_argument("--dry-run", action="store_true", help="只列出不實際更新")
    parser.add_argument("--user-id", type=int, default=None, help="只更新特定使用者")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, user_id_filter=args.user_id))
