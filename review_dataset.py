#!/usr/bin/env python3
"""
review_dataset.py
客語資料集管理員人工終審腳本（第二道防線）

使用方式：
    python review_dataset.py

功能：
    - 讀取 dataset_storage/metadata.jsonl 中所有 review_status == "pending" 的資料
    - 逐筆顯示，讓管理員選擇：
        [1] 採用 AI 建議
        [2] 採用初學者填寫
        [3] 手動輸入正確客語字
        [4] 拒絕並刪除這筆垃圾資料
    - 審核通過的資料：
        * review_status → "verified"
        * 新增 gold_text（最終黃金標準）
        * 附加至 dataset_storage/verified_metadata.jsonl
    - 拒絕的資料：
        * review_status → "rejected"
    - 最後將所有資料（含已更新狀態）回寫至 metadata.jsonl
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
DATASET_DIR      = Path("dataset_storage")
METADATA_FILE    = DATASET_DIR / "metadata.jsonl"
VERIFIED_FILE    = DATASET_DIR / "verified_metadata.jsonl"

# ── ANSI 顏色（Windows cmd 若不支援可關掉）────────────────────────────────────
try:
    import os
    # 在 Windows 上啟用 ANSI 支援
    if os.name == "nt":
        os.system("")   # 觸發 Windows 10+ 的 ANSI 模式
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except Exception:
    RED = GREEN = YELLOW = CYAN = BOLD = RESET = ""


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def load_all_records() -> list[dict]:
    """讀取 metadata.jsonl 全部資料，忽略空行與格式錯誤行。"""
    if not METADATA_FILE.exists():
        return []
    records = []
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"{YELLOW}⚠️  第 {lineno} 行 JSON 解析失敗，已跳過：{e}{RESET}")
    return records


def save_all_records(records: list[dict]) -> None:
    """將所有資料回寫至 metadata.jsonl（覆蓋）。"""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_verified(record: dict) -> None:
    """將一筆已審核資料附加至 verified_metadata.jsonl。"""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with VERIFIED_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_divider() -> None:
    print(f"{CYAN}{'─' * 54}{RESET}")


def display_record(index: int, total: int, rec: dict) -> None:
    """在終端機以易讀格式顯示一筆待審資料。"""
    print()
    print_divider()
    print(f"{BOLD}  [{index}/{total}]  ID: {rec.get('id', 'N/A')}{RESET}")
    print(f"  情境：{YELLOW}{rec.get('scenario', 'N/A')}{RESET}")
    print(f"  建立：{rec.get('created_at', 'N/A')}")
    print_divider()
    print(f"  {RED}❌ 原始誤判：   {rec.get('wrong_text', '')}{RESET}")
    print(f"  👤 初學者填寫：{YELLOW}{rec.get('correct_text', '')}{RESET}")
    print(f"  🤖 AI 預審建議：{GREEN}{rec.get('ai_suggested_text', '（無）')}{RESET}")
    print_divider()


def prompt_choice() -> str:
    """顯示選項並取得管理員輸入，直到輸入合法為止。"""
    print(f"\n  {BOLD}請選擇操作：{RESET}")
    print(f"    {GREEN}[1]{RESET} 採用 AI 建議")
    print(f"    {YELLOW}[2]{RESET} 採用初學者填寫")
    print(f"    {CYAN}[3]{RESET} 手動輸入正確客語字")
    print(f"    {RED}[4]{RESET} 拒絕並刪除（垃圾資料）")
    print(f"    {BOLD}[q]{RESET} 暫停審核，儲存進度後離開")
    while True:
        choice = input("\n  > ").strip().lower()
        if choice in ("1", "2", "3", "4", "q"):
            return choice
        print(f"  {RED}無效輸入，請輸入 1 / 2 / 3 / 4 / q{RESET}")


def prompt_manual_text() -> str:
    """讓管理員手動輸入正確客語字，不可為空。"""
    while True:
        text = input(f"  {CYAN}請輸入正確的客語字：{RESET}").strip()
        if text:
            return text
        print(f"  {RED}不可為空，請重新輸入。{RESET}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{GREEN}╔══════════════════════════════════════════════╗")
    print(f"║   客語資料集管理員人工終審系統（第二道防線）  ║")
    print(f"╚══════════════════════════════════════════════╝{RESET}\n")

    # 讀取全部資料
    all_records = load_all_records()
    if not all_records:
        print(f"{YELLOW}⚠️  metadata.jsonl 不存在或為空，無資料可審核。{RESET}")
        sys.exit(0)

    # 篩出待審資料（保留原始 index 以便回寫）
    pending_indices = [
        i for i, r in enumerate(all_records)
        if r.get("review_status") == "pending"
    ]

    if not pending_indices:
        print(f"{GREEN}✅ 目前沒有待審核的資料，所有資料均已處理完畢。{RESET}")
        sys.exit(0)

    total   = len(pending_indices)
    done    = 0
    verified_count = 0
    rejected_count = 0

    print(f"  共找到 {BOLD}{total}{RESET} 筆待審資料，開始逐筆審核...\n")

    for seq, orig_idx in enumerate(pending_indices, 1):
        rec = all_records[orig_idx]
        display_record(seq, total, rec)

        choice = prompt_choice()

        # ── [q] 暫停 ──────────────────────────────────────────────────────────
        if choice == "q":
            print(f"\n  {YELLOW}⏸  已暫停審核，儲存目前進度...{RESET}")
            break

        # ── 決定 gold_text ────────────────────────────────────────────────────
        if choice == "1":
            gold_text = rec.get("ai_suggested_text") or rec.get("correct_text", "")
            source    = "ai_suggested"
        elif choice == "2":
            gold_text = rec.get("correct_text", "")
            source    = "user_input"
        elif choice == "3":
            gold_text = prompt_manual_text()
            source    = "admin_manual"
        else:  # choice == "4"
            # 拒絕：標記 rejected，不寫入 verified
            all_records[orig_idx]["review_status"] = "rejected"
            all_records[orig_idx]["reviewed_at"]   = datetime.now(timezone.utc).isoformat()
            rejected_count += 1
            done += 1
            print(f"  {RED}🗑  已標記為拒絕（rejected）。{RESET}")
            continue

        # ── 審核通過：更新欄位 ────────────────────────────────────────────────
        all_records[orig_idx].update({
            "review_status": "verified",
            "gold_text":     gold_text,
            "gold_source":   source,        # 記錄黃金字來源，方便後續分析
            "reviewed_at":   datetime.now(timezone.utc).isoformat(),
        })

        # 附加至 verified_metadata.jsonl
        append_verified(all_records[orig_idx])
        verified_count += 1
        done += 1

        print(f"  {GREEN}✅ 已驗證，黃金文字：「{gold_text}」（來源：{source}）{RESET}")

    # ── 回寫 metadata.jsonl（保留所有資料，狀態已更新）────────────────────────
    save_all_records(all_records)

    # ── 結果摘要 ──────────────────────────────────────────────────────────────
    remaining = total - done
    print()
    print_divider()
    print(f"{BOLD}  審核完成摘要{RESET}")
    print_divider()
    print(f"  {GREEN}✅ 已驗證（verified）：{verified_count} 筆{RESET}")
    print(f"  {RED}🗑  已拒絕（rejected）：{rejected_count} 筆{RESET}")
    if remaining > 0:
        print(f"  {YELLOW}⏸  尚未審核（pending）：{remaining} 筆{RESET}")
    print_divider()
    print(f"\n  黃金資料集已儲存至：{BOLD}{VERIFIED_FILE}{RESET}")
    print(f"  完整記錄已回寫至：  {BOLD}{METADATA_FILE}{RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}⚠️  使用者中斷（Ctrl+C），進度未儲存。{RESET}\n")
        sys.exit(1)
