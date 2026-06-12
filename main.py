from contextlib import asynccontextmanager
import pathlib
from pathlib import Path
from dotenv import load_dotenv

# 必須在所有其他 import 之前載入 .env，確保 DATABASE_URL 等環境變數正確
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from fastapi import FastAPI, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text

from database import get_db, engine, Base
from models import User

from routers.profile import router as profile_router
from routers.auth import router as auth_router
from routers.recognition import router as recognition_router
from routers.learning import router as learning_router, init_llm
from routers.practice import router as practice_router
from routers.saved_words import router as saved_words_router
from routers.hakka_api import router as hakka_router
from routers.ocr import router as ocr_router
from routers.dataset import router as dataset_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text(
                "ALTER TABLE saved_words "
                "ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'yolo'"
            ))
        print("DB tables ready")
    except Exception as e:
        print(f"DB init warning: {e}")

    # 在 .env 確定載入後初始化 LLM
    try:
        init_llm()
    except Exception as e:
        print(f"[LLM] 初始化失敗（不影響啟動）: {e}")

    # 預設練習音檔：若不存在則自動用 TTS API 產生
    try:
        from routers.practice import ensure_preset_audios
        await ensure_preset_audios()
    except Exception as e:
        print(f"[Practice] 預設音檔初始化失敗（不影響啟動）: {e}")

    print("App started")
    yield
    print("App stopped")

app = FastAPI(lifespan=lifespan)

# 開發模式：所有 HTML 頁面禁止快取，確保每次都拿到最新版本
@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    response = await call_next(request)
    # HTML 頁面 + 靜態 JS/CSS 全部禁止快取
    path = request.url.path
    if (path.endswith(".html") or path.endswith(".js") or path.endswith(".css")
            or path in ("/", "/login", "/profile", "/recognition",
                        "/practice", "/learning", "/game", "/record")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Static files
pathlib.Path("static/uploads").mkdir(parents=True, exist_ok=True)
pathlib.Path("static/audios/words").mkdir(parents=True, exist_ok=True)
pathlib.Path("static/audios/sentences").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


# Voice practice files
_vp_audios = pathlib.Path("voice_practice/data/audios")
_vp_images = pathlib.Path("voice_practice/data/images")

_vp_audios.mkdir(parents=True, exist_ok=True)
_vp_images.mkdir(parents=True, exist_ok=True)

app.mount("/voice_practice/audios", StaticFiles(directory=str(_vp_audios)), name="vp_audios")
app.mount("/voice_practice/images", StaticFiles(directory=str(_vp_images)), name="vp_images")


# Routers
app.include_router(profile_router)
app.include_router(auth_router)
app.include_router(recognition_router)
app.include_router(learning_router)
app.include_router(practice_router)
app.include_router(saved_words_router)
app.include_router(hakka_router)
app.include_router(ocr_router)
app.include_router(dataset_router)


# Serve pages
@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")


@app.get("/profile")
async def profile_page():
    return FileResponse("static/profile.html")


@app.get("/recognition")
async def recognition_page():
    return FileResponse("static/recognition.html")


@app.get("/practice")
async def practice_page():
    return FileResponse("static/practice.html")


@app.get("/learning")
async def learning_page():
    return FileResponse("static/learning.html")


@app.get("/game")
async def game_page():
    return FileResponse("static/game.html")


@app.get("/record")
async def record_page():
    return FileResponse("static/record.html")


# Existing endpoints
@app.get("/users")
async def read_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    return result.scalars().all()


@app.post("/users")
async def create_user(name: str, email: str, db: AsyncSession = Depends(get_db)):
    new_user = User(name=name, email=email)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user


@app.get("/test-db")
async def test_db():
    return {"status": "ok"}
