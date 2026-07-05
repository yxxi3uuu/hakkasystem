# 客語隨拍隨學 系統啟動說明

## 一、第一次環境設定（只需做一次）

### Step 1：建立 Python 3.11 虛擬環境

```bash
conda create -n hakka311 python=3.11 -y
conda activate hakka311
```

### Step 2：安裝所有套件

```bash
pip install -r requirements.txt
```

### Step 3：安裝 LLM 套件（Windows 專用）

```bash
pip uninstall llama-cpp-python llama-cpp-python-bundled -y
pip install llama-cpp-python==0.2.90 --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> 如果出現 `WinError 127`，請安裝 [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) 後重開機再試。

### Step 4：設定環境變數

```bash
copy .env.example .env
```

用文字編輯器打開 `.env`，填入：
- `HAKKA_USERNAME` / `HAKKA_PASSWORD`：客語 API 帳號
- `GEMINI_API_KEY`：Google Gemini API 金鑰（從 https://aistudio.google.com/apikey 取得）
- `LLM_MODEL_PATH`：模型路徑（預設 `data/qwen2.5-3b-instruct-q4_k_m.gguf`）

### Step 5：放入模型檔

把 `qwen2.5-3b-instruct-q4_k_m.gguf` 放到專案的 `data/` 資料夾。

### Step 6：啟動 Docker 資料庫

```bash
docker-compose up -d db
```

---

## 二、每次啟動系統

```bash
conda activate hakka311
python -m uvicorn main:app --reload
```

開啟瀏覽器：http://localhost:8000

---

## 三、預設帳號

| 帳號 | 角色 | 預設密碼 |
|------|------|----------|
| s0903057896@gmail.com | 超級管理者 | Hakka2026 |
| 112707530@cc.ncu.edu.tw | 老師 | Hakka2026 |
| roylin915@gmail.com | 老師 | Hakka2026 |
| lys20050214@gmail.com | 老師 | Hakka2026 |
| justin0516@g.ncu.edu.tw | 老師 | Hakka2026 |

> 第一次登入後請至「設定」修改密碼。

---

## 四、更新舊資料的 LLM 句子（選用）

如果資料庫有舊的 fallback 句子（`這是一個關於XX的生活句子`），可執行：

```bash
conda activate hakka311
# 預覽要更新哪些（不實際修改）
python update_sentences.py --dry-run

# 更新自己的資料（user_id=1 為例）
python update_sentences.py --user-id 1

# 更新所有人的資料
python update_sentences.py
```

---

## 五、Docker 完整啟動（不需要本機 Python）

```bash
docker-compose up --build -d
```

開啟瀏覽器：http://localhost:8000

> 注意：Docker 模式下 LLM 不可用（需要把模型檔掛載到容器）。

---

## 六、部署到雲端

### Railway（推薦）
1. 推上 GitHub
2. 至 [railway.app](https://railway.app) 用 GitHub 登入
3. New Project → Deploy from GitHub
4. 加 PostgreSQL service
5. 設定環境變數（`.env` 的內容）

### Render
1. 至 [render.com](https://render.com)
2. New Web Service → 連 GitHub repo
3. 加 PostgreSQL database
4. 設環境變數
