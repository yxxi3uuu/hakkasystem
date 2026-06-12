from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from pydantic import BaseModel
from database import get_db
from models import User, LearningStats, Achievement, WeeklyGoal, SavedWord, Activity
import datetime

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

class ActivityCreateRequest(BaseModel):
    icon: str
    title: str
    score: int

# ── 預設成就資料 ──────────────────────────────────────
DEFAULT_ACHIEVEMENTS = [
    {"name": "初學者",   "icon": "🌱", "description": "完成第一次學習"},
    {"name": "勤學獎",   "icon": "📚", "description": "連續學習 7 天"},
    {"name": "發音達人", "icon": "🎤", "description": "語音練習得分 90 分以上 10 次"},
    {"name": "遊戲高手", "icon": "🎮", "description": "配對遊戲 5 步內完成"},
    {"name": "詞彙王",   "icon": "👑", "description": "學習 100 個詞彙"},
    {"name": "完美主義", "icon": "⭐", "description": "連續 5 次得分 100 分"},
]

# ── 預設本週目標 ──────────────────────────────────────
DEFAULT_WEEKLY_GOALS = [
    {"title": "本週拍照學習",   "target": 5},
    {"title": "本週語音練習",   "target": 10},
    {"title": "本週學習天數",   "target": 5},
]

# ── 取得本週起始日（週一）──────────────────────────────
def get_week_start() -> str:
    today = datetime.date.today()
    monday = today - datetime.timedelta(days=today.weekday())
    return monday.strftime("%Y-%m-%d")

# ── Endpoints ─────────────────────────────────────────
@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: int, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    vocab_count_res = await db.execute(select(func.count(SavedWord.id)).where(SavedWord.user_id == user_id))
    vocab_count = vocab_count_res.scalar()

    activities_res = await db.execute(select(Activity).where(Activity.user_id == user_id).order_by(Activity.id.desc()))
    activities = activities_res.scalars().all()

    unique_days = len(set([a.created_at for a in activities])) if activities else 0

    scored_activities = [a for a in activities if a.score > 0]
    avg_score = sum([a.score for a in scored_activities]) / len(scored_activities) if scored_activities else 0.0

    return ProfileResponse(
        user_id=user.id,
        name=user.name,
        study_days=unique_days,
        vocab_learned=vocab_count,
        avg_score=round(avg_score, 1),
    )

@router.get("/profile/{user_id}/achievements", response_model=list[AchievementResponse])
async def get_achievements(user_id: int, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    rows = (await db.execute(
        select(Achievement).where(Achievement.user_id == user_id)
    )).scalars().all()

    if not rows:
        rows = [
            Achievement(user_id=user_id, unlocked=False, **a)
            for a in DEFAULT_ACHIEVEMENTS
        ]
        db.add_all(rows)
        await db.commit()
        for r in rows:
            await db.refresh(r)

    # ── 根據實際數據自動解鎖成就 ──
    activities_res = await db.execute(select(Activity).where(Activity.user_id == user_id))
    activities = activities_res.scalars().all()

    vocab_count_res = await db.execute(select(func.count(SavedWord.id)).where(SavedWord.user_id == user_id))
    vocab_count = vocab_count_res.scalar() or 0

    total_activities = len(activities)
    unique_days = len(set([a.created_at for a in activities]))
    high_scores = [a for a in activities if a.score >= 90]
    perfect_scores = [a for a in activities if a.score == 100]

    # 計算連續 100 分次數
    consecutive_100 = 0
    max_consecutive_100 = 0
    for a in sorted(activities, key=lambda x: x.id):
        if a.score == 100:
            consecutive_100 += 1
            max_consecutive_100 = max(max_consecutive_100, consecutive_100)
        else:
            consecutive_100 = 0

    unlock_map = {
        "初學者":   total_activities >= 1,
        "勤學獎":   unique_days >= 7,
        "發音達人": len(high_scores) >= 10,
        "遊戲高手": any("遊戲" in a.title for a in activities),
        "詞彙王":   vocab_count >= 100,
        "完美主義": max_consecutive_100 >= 5,
    }

    changed = False
    for row in rows:
        should_unlock = unlock_map.get(row.name, False)
        if should_unlock and not row.unlocked:
            row.unlocked = True
            changed = True

    if changed:
        await db.commit()
        for r in rows:
            await db.refresh(r)

    return rows

@router.get("/profile/{user_id}/weekly-goals", response_model=list[WeeklyGoalResponse])
async def get_weekly_goals(user_id: int, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    rows = (await db.execute(
        select(WeeklyGoal).where(WeeklyGoal.user_id == user_id)
    )).scalars().all()

    # 若沒有目標，自動建立預設目標
    if not rows:
        rows = [
            WeeklyGoal(user_id=user_id, current=0, **g)
            for g in DEFAULT_WEEKLY_GOALS
        ]
        db.add_all(rows)
        await db.commit()
        for r in rows:
            await db.refresh(r)

    # ── 根據本週實際活動更新進度 ──
    week_start = get_week_start()
    activities_res = await db.execute(
        select(Activity)
        .where(Activity.user_id == user_id, Activity.created_at >= week_start)
    )
    week_activities = activities_res.scalars().all()

    # 本週拍照次數
    photo_count = sum(1 for a in week_activities if "拍照" in a.title)
    # 本週語音練習次數
    practice_count = sum(1 for a in week_activities if "語音" in a.title or a.score > 0)
    # 本週學習天數
    study_days = len(set(a.created_at for a in week_activities))

    progress_map = {
        "本週拍照學習": photo_count,
        "本週語音練習": practice_count,
        "本週學習天數": study_days,
    }

    changed = False
    for row in rows:
        new_val = progress_map.get(row.title)
        if new_val is not None and row.current != new_val:
            row.current = new_val
            changed = True

    if changed:
        await db.commit()
        for r in rows:
            await db.refresh(r)

    return rows

@router.get("/profile/{user_id}/records")
async def get_records(user_id: int, db: AsyncSession = Depends(get_db)):
    activities_res = await db.execute(select(Activity).where(Activity.user_id == user_id).order_by(Activity.id.desc()).limit(10))
    activities = activities_res.scalars().all()

    scored_res = await db.execute(select(Activity).where(Activity.user_id == user_id, Activity.score > 0).order_by(Activity.id.asc()))
    scored_activities = scored_res.scalars().all()
    last_7_scores = [a.score for a in scored_activities[-7:]]
    while len(last_7_scores) < 7:
        last_7_scores.insert(0, 0)

    return {
        "recent_activities": [{"icon": a.icon, "title": a.title, "score": a.score, "date": a.created_at} for a in activities],
        "weekly_scores": last_7_scores
    }

@router.post("/profile/{user_id}/activity")
async def add_activity(user_id: int, req: ActivityCreateRequest, db: AsyncSession = Depends(get_db)):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    new_activity = Activity(
        user_id=user_id,
        icon=req.icon,
        title=req.title,
        score=req.score,
        created_at=today
    )
    db.add(new_activity)
    await db.commit()
    return {"status": "success"}
