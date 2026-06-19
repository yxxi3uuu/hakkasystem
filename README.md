## 啟動方式

1. 安裝依賴

```bash
pip install -r requirements.txt
```

2. 複製 `.env.example` 成 `.env`，再依自己的機器修改值。

3. 本機開發

```bash
uvicorn main:app --reload
```

4. Docker 啟動

```bash
docker-compose up --build
```

## LLM 設定

- 本機執行時，`LLM_MODEL_PATH` 建議設成 `data/qwen2.5-3b-instruct-q4_k_m.gguf`
- Docker 容器內會自動覆蓋成 `/app/data/qwen2.5-3b-instruct-q4_k_m.gguf`
- 模型檔要放在專案的 `data/` 資料夾，或掛載到容器的 `/app/data/`

## 驗證

```text
http://127.0.0.1:8000/test-db
```

應該回傳：

```json
{
  "db": "ok"
}
```

```text
http://127.0.0.1:8000/docs
```

