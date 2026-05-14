from database import Base
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)
    name     = Column(String)
    email    = Column(String, unique=True)
    password = Column(String)  # hashed password

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
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"))
    image_path  = Column(String)
    label_zh    = Column(String)
    label_hakka = Column(String)

class Activity(Base):
    __tablename__ = "activities"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"))
    icon        = Column(String)
    title       = Column(String)
    score       = Column(Integer, default=0)
    created_at  = Column(String) # YYYY-MM-DD


