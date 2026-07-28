from database import Base
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)
    name     = Column(String)
    email    = Column(String, unique=True)
    password = Column(String)  # hashed password
    is_admin = Column(Boolean, default=False)
    role     = Column(String, default="student")  # student | teacher | admin
    avatar_path = Column(String, default="")


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
    word_id = Column(Integer, ForeignKey("certification_words.word_id"), nullable=True)
    recognition_id = Column(Integer, ForeignKey("recognitions.recognition_id"), nullable=True)
    dialect = Column(String, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

class Activity(Base):
    __tablename__ = "activities"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"))
    icon        = Column(String)
    title       = Column(String)
    score       = Column(Integer, default=0)
    created_at  = Column(String) # YYYY-MM-DD


class VocabularySource(Base):
    __tablename__ = "vocabulary_sources"

    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String, nullable=False)
    source_version = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    checksum = Column(String, nullable=False, unique=True)
    import_report_json = Column(Text, default="")
    imported_at = Column(DateTime(timezone=True), server_default=func.now())


class CertificationWord(Base):
    __tablename__ = "certification_words"
    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "source_word_code",
            "dialect",
            name="uq_certification_word_source_code_dialect",
        ),
    )

    word_id = Column(Integer, primary_key=True, index=True)
    source_id = Column(
        Integer,
        ForeignKey("vocabulary_sources.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    source_word_code = Column(String, nullable=False)
    hakka_word = Column(String, nullable=False, index=True)
    zh_meaning = Column(Text, nullable=False)
    dialect = Column(String, nullable=False, index=True)
    pinyin = Column(String, nullable=False)
    certification_level = Column(String, nullable=False, index=True)
    category_code = Column(String, default="")
    category = Column(String, default="", index=True)
    subcategory = Column(String, default="")
    part_of_speech_primary = Column(String, default="")
    part_of_speech_secondary = Column(String, default="")
    example_sentence = Column(Text, default="")
    example_translation = Column(Text, default="")
    audio_url = Column(String, default="")
    notes = Column(Text, default="")
    source = Column(String, nullable=False)
    source_version = Column(String, nullable=False)
    visualizable_type = Column(String, default="")
    is_verified = Column(Boolean, default=True, nullable=False)
    normalization_status = Column(String, default="imported", nullable=False)
    missing_fields_json = Column(Text, default="[]")
    raw_data_json = Column(Text, default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class WordAlias(Base):
    __tablename__ = "word_aliases"
    __table_args__ = (
        UniqueConstraint("alias_text", "word_id", name="uq_word_alias_text_word"),
    )

    alias_id = Column(Integer, primary_key=True, index=True)
    alias_text = Column(String, nullable=False, index=True)
    canonical_zh_name = Column(String, nullable=False, index=True)
    word_id = Column(
        Integer,
        ForeignKey("certification_words.word_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    confidence = Column(Float, default=0.0, nullable=False)
    confirmed_count = Column(Integer, default=0, nullable=False)
    correction_count = Column(Integer, default=0, nullable=False)
    source = Column(String, default="")
    is_verified = Column(Boolean, default=False, nullable=False)


class WordRelation(Base):
    __tablename__ = "word_relations"
    __table_args__ = (
        UniqueConstraint(
            "source_word_id",
            "target_word_id",
            "relation_type",
            name="uq_word_relation_edge_type",
        ),
    )

    relation_id = Column(Integer, primary_key=True, index=True)
    source_word_id = Column(
        Integer,
        ForeignKey("certification_words.word_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_word_id = Column(
        Integer,
        ForeignKey("certification_words.word_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relation_type = Column(String, nullable=False, index=True)
    relation_score = Column(Float, default=0.0, nullable=False)
    source = Column(String, default="")
    is_verified = Column(Boolean, default=False, nullable=False)


class Recognition(Base):
    __tablename__ = "recognitions"

    recognition_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    anonymous_user_id = Column(String, default="", index=True)
    image_path = Column(String, default="")
    original_vlm_result = Column(Text, default="{}")
    normalized_result = Column(Text, default="{}")
    final_word_id = Column(
        Integer,
        ForeignKey("certification_words.word_id", ondelete="SET NULL"),
        nullable=True,
    )
    confidence = Column(Float, default=0.0)
    user_confirmed = Column(Boolean, default=False)
    corrected_word_id = Column(
        Integer,
        ForeignKey("certification_words.word_id", ondelete="SET NULL"),
        nullable=True,
    )
    scene = Column(String, default="")
    response_time_ms = Column(Integer, default=0)
    model_version = Column(String, default="")
    prompt_version = Column(String, default="")
    system_version = Column(String, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserWordProgress(Base):
    __tablename__ = "user_word_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "word_id", name="uq_user_word_progress"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    word_id = Column(
        Integer,
        ForeignKey("certification_words.word_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    viewed_count = Column(Integer, default=0, nullable=False)
    practice_count = Column(Integer, default=0, nullable=False)
    correct_count = Column(Integer, default=0, nullable=False)
    incorrect_count = Column(Integer, default=0, nullable=False)
    last_learned_at = Column(DateTime(timezone=True), nullable=True)
    mastery_status = Column(String, default="unlearned", nullable=False)
    next_review_at = Column(DateTime(timezone=True), nullable=True)
