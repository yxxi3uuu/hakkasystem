pip install "fastapi[standard]"
安裝fastapi

安裝docker desktop 
docker-compose up --build

docker ps
看有沒有聯繫成功

uvicorn main:app --reload
會出來一個網址點下去應該有
http://127.0.0.1:8000/test-db 看資料庫有沒有連上
{
  "db": "ok"
}

http://127.0.0.1:8000/docs 看後端有沒有開啟

