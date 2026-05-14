from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db, engine, Base
from models import User
from routers.profile import router as profile_router
from routers.auth import router as auth_router
from routers.recognition import router as recognition_router
from routers.learning import router as learning_router
from routers.practice import router as practice_router
from routers.saved_words import router as saved_words_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("DB tables ready")
    except Exception as e:
        print(f"DB init warning: {e}")
    print("App started")
    yield
    print("App stopped")

app = FastAPI(lifespan=lifespan)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# 確保 voice_practice 目錄存在再掛載
import pathlib
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
def test_db():
    return {"status": "ok"}
