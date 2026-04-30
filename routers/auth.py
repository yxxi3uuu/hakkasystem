from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel, EmailStr
from database import get_db
from models import User
import hashlib
import secrets

router = APIRouter(prefix="/api/auth")

# ── 簡易密碼 hash（正式環境請用 bcrypt）──
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password: str, stored: str) -> bool:
    salt, hashed = stored.split(":")
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed

# ── Schemas ──
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    user_id: int
    name: str
    email: str
    message: str

# ── 註冊 ──
@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # 檢查 email 是否已存在
    existing = (await db.execute(
        select(User).where(User.email == req.email)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="此 Email 已被註冊")

    user = User(
        name=req.name,
        email=req.email,
        password=hash_password(req.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return AuthResponse(
        user_id=user.id,
        name=user.name,
        email=user.email,
        message="註冊成功",
    )

# ── 登入 ──
@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(
        select(User).where(User.email == req.email)
    )).scalar_one_or_none()

    if not user or not user.password:
        raise HTTPException(status_code=401, detail="Email 或密碼錯誤")

    if not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="Email 或密碼錯誤")

    return AuthResponse(
        user_id=user.id,
        name=user.name,
        email=user.email,
        message="登入成功",
    )

# ── 登出 ──
@router.post("/logout")
async def logout():
    return {"message": "已登出"}
