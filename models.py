from database import Base
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)
    name     = Column(String)
    email    = Column(String, unique=True)
    password = Column(String)  # hashed password
    is_admin = Column(Boolean, default=False)
    role     = Column(String, default="student")  # student | teacher | admin


class Class(Base):
    __tablename__ = "classes"
    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String, nullable=False)
    description = Column(String, default="")


class ClassTeacher(Base):
    __tablename__ = "class_teachers"
    id         = Column(Integer, primary_key=True, index=True)
    class_id   = Column(Integer, ForeignKey("classes.id", ondelete="CASCADE"))
    teacher_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))


class ClassStudent(Base):
    __tablename__ = "class_students"
    id         = Column(Integer, primary_key=True, index=True)
    class_id   = Column(Integer, ForeignKey("classes.id", ondelete="CASCADE"))
    student_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))


class Notification(Base):
    __tablename__ = "notifications"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))  # 收件人
    sender_id  = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title      = Column(String, nullable=False)
    body       = Column(String, default="")
    is_read    = Column(Boolean, default=False)
    created_at = Column(String)  # YYYY-MM-DD HH:MM

class LearningStats(Base):
    __tablename__ = "learning_stats"
    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id"), unique=True)
    vocab_learned = Column(Integer, default=0)
    avg_score     = Column(Float, default=0.0)
    study_days    = Column(Integer, default=0)

class Achievement(Base):
    __tablename__ = "achievements"
    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, ForeignKey("users.id"))
    name        = Column(String)
    icon        = Column(String)
    description = Column(String)
    unlocked    = Column(Boolean, default=False)

class WeeklyGoal(Base):
    __tablename__ = "weekly_goals"
    id      = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title   = Column(String)
    target  = Column(Integer)
    current = Column(Integer, default=0)

class SavedWord(Base):
    __tablename__ = "saved_words"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    image_path = Column(String)

    label_zh = Column(String)
    label_hakka = Column(String)
    label_pinyin = Column(String, default="")
    audio_path = Column(String, default="")

    labels_json = Column(String, default="")

    sentence_zh = Column(String, default="")
    sentence_hakka = Column(String, default="")
    sentence_audio_path = Column(String, default="")

    source = Column(String, default="yolo")   # yolo | ocr

class Activity(Base):
    __tablename__ = "activities"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"))
    icon        = Column(String)
    title       = Column(String)
    score       = Column(Integer, default=0)
    created_at  = Column(String) # YYYY-MM-DD


