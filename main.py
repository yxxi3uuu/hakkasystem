from contextlib import asynccontextmanager
import pathlib
from pathlib import Path
from dotenv import load_dotenv

# 必須在所有其他 import 之前載入 .env，確保 DATABASE_URL 等環境變數正確
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

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
from routers.admin import router as admin_router
from routers.notification import router as notification_router

# ── 預設帳號設定 ──────────────────────────────────────────────────────
_SUPER_ADMIN   = {"name": "吳怡臻", "email": "s0903057896@gmail.com", "role": "admin"}
_DEFAULT_TEACHERS = [
    {"name": "黃璿羽", "email": "112707530@cc.ncu.edu.tw", "role": "teacher"},
    {"name": "林守毅", "email": "roylin915@gmail.com",      "role": "teacher"},
    {"name": "梁易軒", "email": "lys20050214@gmail.com",    "role": "teacher"},
    {"name": "張育倫", "email": "justin0516@g.ncu.edu.tw",  "role": "teacher"},
]
_DEFAULT_PASSWORD = "Hakka2026"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("DB tables ready")
    except Exception as e:
        print(f"DB init warning: {e}")

    # ── 自動 migration（補齊新欄位和新表格）──────────────────────────
    try:
        from database import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as db:
            migrations = [
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE",
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR DEFAULT 'student'",
                "UPDATE users SET role='admin' WHERE is_admin=TRUE AND role='student'",
                "ALTER TABLE saved_words ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'yolo'",
                "ALTER TABLE saved_words ADD COLUMN IF NOT EXISTS label_pinyin VARCHAR DEFAULT ''",
                "ALTER TABLE saved_words ADD COLUMN IF NOT EXISTS sentence_audio_path VARCHAR DEFAULT ''",
                "ALTER TABLE saved_words ADD COLUMN IF NOT EXISTS labels_json VARCHAR DEFAULT ''",
                """CREATE TABLE IF NOT EXISTS classes (
                    id SERIAL PRIMARY KEY, name VARCHAR NOT NULL, description VARCHAR DEFAULT '')""",
                """CREATE TABLE IF NOT EXISTS class_teachers (
                    id SERIAL PRIMARY KEY,
                    class_id INTEGER REFERENCES classes(id) ON DELETE CASCADE,
                    teacher_id INTEGER REFERENCES users(id) ON DELETE CASCADE)""",
                """CREATE TABLE IF NOT EXISTS class_students (
                    id SERIAL PRIMARY KEY,
                    class_id INTEGER REFERENCES classes(id) ON DELETE CASCADE,
                    student_id INTEGER REFERENCES users(id) ON DELETE CASCADE)""",
                """CREATE TABLE IF NOT EXISTS notifications (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    sender_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    title VARCHAR NOT NULL, body VARCHAR DEFAULT '',
                    is_read BOOLEAN DEFAULT FALSE, created_at VARCHAR)""",
            ]
            for sql in migrations:
                try:
                    await db.execute(text(sql))
                except Exception:
                    pass
            await db.commit()
            print("DB migration done")
    except Exception as e:
        print(f"DB migration warning: {e}")

    # 確保預設帳號存在且 role 正確
    try:
        from routers.auth import hash_password
        from database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            for account in [_SUPER_ADMIN] + _DEFAULT_TEACHERS:
                existing = (await db.execute(
                    select(User).where(User.email == account["email"])
                )).scalar_one_or_none()
                if existing:
                    if existing.role != account["role"]:
                        existing.role = account["role"]
                        existing.is_admin = (account["role"] == "admin")
                        await db.commit()
                        print(f"[Init] 已更新角色 {account['email']} → {account['role']}")
                else:
                    db.add(User(
                        name=account["name"],
                        email=account["email"],
                        password=hash_password(_DEFAULT_PASSWORD),
                        role=account["role"],
                        is_admin=(account["role"] == "admin"),
                    ))
                    await db.commit()
                    print(f"[Init] 已建立帳號 {account['email']}（role={account['role']}，密碼：{_DEFAULT_PASSWORD}）")
    except Exception as e:
        print(f"[Init] 初始化帳號失敗：{e}")

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

    # YOLO 模型背景預載：在伺服器啟動時就載入，避免第一次 request 卡住
    import asyncio as _asyncio

    async def _preload_yolo():
        try:
            await _asyncio.to_thread(
                lambda: __import__('routers.recognition', fromlist=['_get_yolo_model'])._get_yolo_model()
            )
        except Exception as e:
            print(f"[YOLO] 背景預載失敗（不影響啟動）: {e}")

    _asyncio.ensure_future(_preload_yolo())
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
app.include_router(admin_router)
app.include_router(notification_router)


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


@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html")


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

