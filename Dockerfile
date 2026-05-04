# 使用 Python 3.10 輕量版
FROM python:3.10-slim

# 設定容器內的工作目錄
WORKDIR /app

# 複製套件清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼到容器
COPY . .

# 啟動 FastAPI (使用 uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

