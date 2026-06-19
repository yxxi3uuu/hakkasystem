"""
routers/admin.py
管理者後台 API
- admin：可看全部使用者、管理老師/班級
- teacher：只能看自己班級的學生
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_
from pydantic import BaseModel

from database import get_db
from models import User, SavedWord, Activity, Class, ClassTeacher, ClassStudent

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── 權限驗證 ──────────────────────────────────────────────────────────
async def require_teacher_or_admin(
    admin_id: int = Query(...),
    db: AsyncSession = Depends(get_db)
) -> User:
    user = (await db.execute(select(User).where(User.id == admin_id))).scalar_one_or_none()
    if not user or getattr(user, "role", "student") not in ("teacher", "admin"):
        raise HTTPException(status_code=403, detail="需要老師或管理者權限")
    return user


async def require_admin_only(
    admin_id: int = Query(...),
    db: AsyncSession = Depends(get_db)
) -> User:
    user = (await db.execute(select(User).where(User.id == admin_id))).scalar_one_or_none()
    if not user or getattr(user, "role", "student") != "admin":
        raise HTTPException(status_code=403, detail="需要管理者權限")
    return user


# ── 共用：取得使用者統計資料 ──────────────────────────────────────────
async def _user_stats(user_id: int, db: AsyncSession) -> dict:
    vocab_count = (await db.execute(
        select(func.count(SavedWord.id)).where(SavedWord.user_id == user_id)
    )).scalar() or 0

    activity_count = (await db.execute(
        select(func.count(Activity.id)).where(Activity.user_id == user_id)
    )).scalar() or 0

    scored = (await db.execute(
        select(Activity).where(Activity.user_id == user_id, Activity.score > 0)
    )).scalars().all()
    avg_score = round(sum(a.score for a in scored) / len(scored), 1) if scored else 0.0

    return {
        "vocab_count": vocab_count,
        "activity_count": activity_count,
        "avg_score": avg_score,
    }


# ════════════════════════════════════════════════════════════════════════
# 使用者管理（admin only）
# ════════════════════════════════════════════════════════════════════════

@router.get("/users")
async def list_all_users(
    caller: User = Depends(require_admin_only),
    db: AsyncSession = Depends(get_db)
):
    users = (await db.execute(select(User))).scalars().all()
    result = []
    for u in users:
        stats = await _user_stats(u.id, db)
        result.append({
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "role": getattr(u, "role", "student") or "student",
            **stats,
        })
    return result


@router.post("/users/{user_id}/set-role")
async def set_user_role(
    user_id: int,
    role: str = Query(..., regex="^(student|teacher|admin)$"),
    caller: User = Depends(require_admin_only),
    db: AsyncSession = Depends(get_db)
):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")
    user.role = role
    user.is_admin = (role == "admin")
    await db.commit()
    return {"status": "success", "user_id": user_id, "role": role}


# ════════════════════════════════════════════════════════════════════════
# 班級管理
# ════════════════════════════════════════════════════════════════════════

class ClassCreate(BaseModel):
    name: str
    description: str = ""


@router.get("/classes")
async def list_classes(
    caller: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    """admin 看全部班級；teacher 只看自己的班級"""
    if getattr(caller, "role", "") == "admin":
        classes = (await db.execute(select(Class))).scalars().all()
    else:
        teacher_classes = (await db.execute(
            select(ClassTeacher).where(ClassTeacher.teacher_id == caller.id)
        )).scalars().all()
        class_ids = [ct.class_id for ct in teacher_classes]
        if not class_ids:
            return []
        classes = (await db.execute(
            select(Class).where(Class.id.in_(class_ids))
        )).scalars().all()

    result = []
    for c in classes:
        student_count = (await db.execute(
            select(func.count(ClassStudent.id)).where(ClassStudent.class_id == c.id)
        )).scalar() or 0
        teachers = (await db.execute(
            select(User).join(ClassTeacher, ClassTeacher.teacher_id == User.id)
            .where(ClassTeacher.class_id == c.id)
        )).scalars().all()
        result.append({
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "student_count": student_count,
            "teachers": [{"id": t.id, "name": t.name} for t in teachers],
        })
    return result


@router.post("/classes")
async def create_class(
    body: ClassCreate,
    caller: User = Depends(require_admin_only),
    db: AsyncSession = Depends(get_db)
):
    new_class = Class(name=body.name, description=body.description)
    db.add(new_class)
    await db.commit()
    await db.refresh(new_class)
    return {"status": "success", "id": new_class.id, "name": new_class.name}


@router.delete("/classes/{class_id}")
async def delete_class(
    class_id: int,
    caller: User = Depends(require_admin_only),
    db: AsyncSession = Depends(get_db)
):
    cls = (await db.execute(select(Class).where(Class.id == class_id))).scalar_one_or_none()
    if not cls:
        raise HTTPException(status_code=404, detail="班級不存在")
    await db.delete(cls)
    await db.commit()
    return {"status": "success"}


# ── 班級成員管理 ────────────────────────────────────────────────────────

@router.post("/classes/{class_id}/teachers/{teacher_id}")
async def add_teacher_to_class(
    class_id: int,
    teacher_id: int,
    caller: User = Depends(require_admin_only),
    db: AsyncSession = Depends(get_db)
):
    existing = (await db.execute(
        select(ClassTeacher).where(
            ClassTeacher.class_id == class_id,
            ClassTeacher.teacher_id == teacher_id
        )
    )).scalar_one_or_none()
    if not existing:
        db.add(ClassTeacher(class_id=class_id, teacher_id=teacher_id))
        await db.commit()
    return {"status": "success"}


@router.delete("/classes/{class_id}/teachers/{teacher_id}")
async def remove_teacher_from_class(
    class_id: int,
    teacher_id: int,
    caller: User = Depends(require_admin_only),
    db: AsyncSession = Depends(get_db)
):
    row = (await db.execute(
        select(ClassTeacher).where(
            ClassTeacher.class_id == class_id,
            ClassTeacher.teacher_id == teacher_id
        )
    )).scalar_one_or_none()
    if row:
        await db.delete(row)
        await db.commit()
    return {"status": "success"}


@router.post("/classes/{class_id}/students/{student_id}")
async def add_student_to_class(
    class_id: int,
    student_id: int,
    caller: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    # teacher 只能操作自己的班級
    if getattr(caller, "role", "") == "teacher":
        ct = (await db.execute(
            select(ClassTeacher).where(
                ClassTeacher.class_id == class_id,
                ClassTeacher.teacher_id == caller.id
            )
        )).scalar_one_or_none()
        if not ct:
            raise HTTPException(status_code=403, detail="無此班級的管理權限")

    existing = (await db.execute(
        select(ClassStudent).where(
            ClassStudent.class_id == class_id,
            ClassStudent.student_id == student_id
        )
    )).scalar_one_or_none()
    if not existing:
        db.add(ClassStudent(class_id=class_id, student_id=student_id))
        await db.commit()
    return {"status": "success"}


@router.delete("/classes/{class_id}/students/{student_id}")
async def remove_student_from_class(
    class_id: int,
    student_id: int,
    caller: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    if getattr(caller, "role", "") == "teacher":
        ct = (await db.execute(
            select(ClassTeacher).where(
                ClassTeacher.class_id == class_id,
                ClassTeacher.teacher_id == caller.id
            )
        )).scalar_one_or_none()
        if not ct:
            raise HTTPException(status_code=403, detail="無此班級的管理權限")

    row = (await db.execute(
        select(ClassStudent).where(
            ClassStudent.class_id == class_id,
            ClassStudent.student_id == student_id
        )
    )).scalar_one_or_none()
    if row:
        await db.delete(row)
        await db.commit()
    return {"status": "success"}


@router.get("/classes/{class_id}/students")
async def list_class_students(
    class_id: int,
    caller: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    if getattr(caller, "role", "") == "teacher":
        ct = (await db.execute(
            select(ClassTeacher).where(
                ClassTeacher.class_id == class_id,
                ClassTeacher.teacher_id == caller.id
            )
        )).scalar_one_or_none()
        if not ct:
            raise HTTPException(status_code=403, detail="無此班級的管理權限")

    student_rows = (await db.execute(
        select(ClassStudent).where(ClassStudent.class_id == class_id)
    )).scalars().all()

    result = []
    for cs in student_rows:
        u = (await db.execute(select(User).where(User.id == cs.student_id))).scalar_one_or_none()
        if not u:
            continue
        stats = await _user_stats(u.id, db)
        result.append({"id": u.id, "name": u.name, "email": u.email, **stats})
    return result


# ════════════════════════════════════════════════════════════════════════
# 個別學生資料（老師可存取自己班級的學生）
# ════════════════════════════════════════════════════════════════════════

async def _check_student_access(caller: User, student_id: int, db: AsyncSession):
    if getattr(caller, "role", "") == "admin":
        return
    # teacher：確認該學生在自己的某個班級
    teacher_classes = (await db.execute(
        select(ClassTeacher).where(ClassTeacher.teacher_id == caller.id)
    )).scalars().all()
    class_ids = [ct.class_id for ct in teacher_classes]
    if not class_ids:
        raise HTTPException(status_code=403, detail="你尚未管理任何班級")
    in_class = (await db.execute(
        select(ClassStudent).where(
            ClassStudent.student_id == student_id,
            ClassStudent.class_id.in_(class_ids)
        )
    )).scalar_one_or_none()
    if not in_class:
        raise HTTPException(status_code=403, detail="此學生不在你的班級")


@router.get("/users/{user_id}/words")
async def get_user_words(
    user_id: int,
    caller: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    await _check_student_access(caller, user_id, db)
    words = (await db.execute(
        select(SavedWord).where(SavedWord.user_id == user_id).order_by(SavedWord.id.desc())
    )).scalars().all()
    return [
        {
            "id": w.id,
            "image_path": w.image_path,
            "label_zh": w.label_zh,
            "label_hakka": w.label_hakka,
            "label_pinyin": getattr(w, "label_pinyin", ""),
            "audio_path": getattr(w, "audio_path", ""),
            "sentence_zh": w.sentence_zh,
            "sentence_hakka": w.sentence_hakka,
            "sentence_audio_path": getattr(w, "sentence_audio_path", ""),
            "source": getattr(w, "source", "yolo"),
        }
        for w in words
    ]


@router.get("/users/{user_id}/activities")
async def get_user_activities(
    user_id: int,
    caller: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    await _check_student_access(caller, user_id, db)
    activities = (await db.execute(
        select(Activity).where(Activity.user_id == user_id).order_by(Activity.id.desc())
    )).scalars().all()
    return [
        {"id": a.id, "icon": a.icon, "title": a.title, "score": a.score, "created_at": a.created_at}
        for a in activities
    ]
