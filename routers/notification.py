"""
routers/notification.py
站內通知系統
- 老師/admin 可發送通知給班級或個人
- 所有登入使用者可查看自己的通知並標記已讀
"""

import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from pydantic import BaseModel

from database import get_db
from models import User, Notification, ClassTeacher, ClassStudent

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


# ── 取得我的通知 ────────────────────────────────────────────────────────
@router.get("")
async def get_my_notifications(user_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(Notification)
        .where(Notification.user_id == user_id)
        .order_by(Notification.id.desc())
    )).scalars().all()

    result = []
    for n in rows:
        sender_name = ""
        if n.sender_id:
            sender = (await db.execute(select(User).where(User.id == n.sender_id))).scalar_one_or_none()
            sender_name = sender.name if sender else ""
        result.append({
            "id": n.id,
            "title": n.title,
            "body": n.body,
            "is_read": n.is_read,
            "created_at": n.created_at,
            "sender_name": sender_name,
        })
    return result


# ── 未讀數量 ────────────────────────────────────────────────────────────
@router.get("/unread-count")
async def get_unread_count(user_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    count = (await db.execute(
        select(func.count(Notification.id))
        .where(Notification.user_id == user_id, Notification.is_read == False)
    )).scalar() or 0
    return {"count": count}


# ── 標記全部已讀 ────────────────────────────────────────────────────────
@router.post("/read-all")
async def mark_all_read(user_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(Notification)
        .where(Notification.user_id == user_id, Notification.is_read == False)
    )).scalars().all()
    for n in rows:
        n.is_read = True
    await db.commit()
    return {"status": "success"}


# ── 標記單筆已讀 ────────────────────────────────────────────────────────
@router.post("/{notif_id}/read")
async def mark_read(notif_id: int, user_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    n = (await db.execute(
        select(Notification).where(Notification.id == notif_id, Notification.user_id == user_id)
    )).scalar_one_or_none()
    if n:
        n.is_read = True
        await db.commit()
    return {"status": "success"}


# ── 刪除通知 ────────────────────────────────────────────────────────────
@router.delete("/{notif_id}")
async def delete_notification(notif_id: int, user_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    n = (await db.execute(
        select(Notification).where(Notification.id == notif_id, Notification.user_id == user_id)
    )).scalar_one_or_none()
    if n:
        await db.delete(n)
        await db.commit()
    return {"status": "success"}


# ── 發送通知（老師/admin）──────────────────────────────────────────────
class SendNotifRequest(BaseModel):
    sender_id: int
    title: str
    body: str = ""
    target: str = "class"   # "class" | "user"
    class_id: int | None = None
    user_id: int | None = None


@router.post("/send")
async def send_notification(req: SendNotifRequest, db: AsyncSession = Depends(get_db)):
    # 驗證 sender 是 teacher 或 admin
    sender = (await db.execute(select(User).where(User.id == req.sender_id))).scalar_one_or_none()
    if not sender or getattr(sender, "role", "student") not in ("teacher", "admin"):
        raise HTTPException(status_code=403, detail="只有老師或管理者可以發送通知")

    if not req.title.strip():
        raise HTTPException(status_code=400, detail="通知標題不可為空")

    recipients: list[int] = []

    if req.target == "class" and req.class_id:
        # 發給整個班級
        if getattr(sender, "role", "") == "teacher":
            ct = (await db.execute(
                select(ClassTeacher).where(
                    ClassTeacher.class_id == req.class_id,
                    ClassTeacher.teacher_id == req.sender_id
                )
            )).scalar_one_or_none()
            if not ct:
                raise HTTPException(status_code=403, detail="無此班級的管理權限")

        students = (await db.execute(
            select(ClassStudent).where(ClassStudent.class_id == req.class_id)
        )).scalars().all()
        recipients = [s.student_id for s in students]

    elif req.target == "user" and req.user_id:
        recipients = [req.user_id]

    elif req.target == "all" and getattr(sender, "role", "") == "admin":
        # admin 可廣播給所有人
        all_users = (await db.execute(select(User))).scalars().all()
        recipients = [u.id for u in all_users if u.id != req.sender_id]

    if not recipients:
        raise HTTPException(status_code=400, detail="找不到收件對象")

    now = _now()
    for uid in recipients:
        db.add(Notification(
            user_id=uid,
            sender_id=req.sender_id,
            title=req.title.strip(),
            body=req.body.strip(),
            is_read=False,
            created_at=now,
        ))
    await db.commit()
    return {"status": "success", "sent_to": len(recipients)}
