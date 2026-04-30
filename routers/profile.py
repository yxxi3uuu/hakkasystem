from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from database import get_db
from models import User, LearningStats, Achievement, WeeklyGoal

router = APIRouter(prefix="/api")

# ── Schemas ──────────────────────────────────────────
class ProfileResponse(BaseModel):
    user_id:       int
    name:          str
    study_days:    int
    vocab_learned: int
    avg_score:     float
    model_config = {"from_attributes": True}

class AchievementResponse(BaseModel):
    id:          int
    name:        str
    icon:        str
    description: str
    unlocked:    bool
    model_config = {"from_attributes": True}

class WeeklyGoalResponse(BaseModel):
    id:      int
    title:   str
    target:  int
    current: int
    model_config = {"from_attributes": True}

# ── 預設成就資料 ──────────────────────────────────────
DEFAULT_ACHIEVEMENTS = [
    {"name": "初學者",   "icon": "🌱", "description": "完成第一次學習"},
    {"name": "勤學獎",   "icon": "📚", "description": "連續學習 7 天"},
    {"name": "發音達人", "icon": "🎤", "description": "語音練習得分 90 分以上 10 次"},
    {"name": "遊戲高手", "icon": "🎮", "description": "配對遊戲 5 步內完成"},
    {"name": "詞彙王",   "icon": "👑", "description": "學習 100 個詞彙"},
    {"name": "完美主義", "icon": "⭐", "description": "連續 5 次得分 100 分"},
]

# ── Endpoints ─────────────────────────────────────────
@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: int, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    stats = (await db.execute(
        select(LearningStats).where(LearningStats.user_id == user_id)
    )).scalar_one_or_none()

    return ProfileResponse(
        user_id=user.id,
        name=user.name,
        study_days=stats.study_days if stats else 0,
        vocab_learned=stats.vocab_learned if stats else 0,
        avg_score=stats.avg_score if stats else 0.0,
    )

@router.get("/profile/{user_id}/achievements", response_model=list[AchievementResponse])
async def get_achievements(user_id: int, db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(Achievement).where(Achievement.user_id == user_id)
    )).scalars().all()

    if not rows:
        # 首次訪問：自動建立預設成就
        rows = [
            Achievement(user_id=user_id, unlocked=False, **a)
            for a in DEFAULT_ACHIEVEMENTS
        ]
        db.add_all(rows)
        await db.commit()
        for r in rows:
            await db.refresh(r)

    return rows

@router.get("/profile/{user_id}/weekly-goals", response_model=list[WeeklyGoalResponse])
async def get_weekly_goals(user_id: int, db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(WeeklyGoal).where(WeeklyGoal.user_id == user_id)
    )).scalars().all()
    return rows
