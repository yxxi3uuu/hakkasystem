import streamlit as st
import pandas as pd
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io
import os

# --- 1. 介面風格深度還原 ---
st.set_page_config(page_title="語音練習", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    .top-header {
        background-color: #5D8A46;
        padding: 20px;
        margin: -60px -100px 20px -100px;
        text-align: center;
        color: white;
    }
    .practice-card {
        background: white;
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #E9ECEF;
    }
    .word-display { font-size: 48px; font-weight: 800; color: #333; margin: 10px 0; }
    .subtitle { color: #666; font-size: 14px; margin-bottom: 20px; text-align: center; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton > button {
        background-color: #5D8A46 !important;
        color: white !important;
        border-radius: 30px !important;
        border: none !important;
        padding: 10px 25px !important;
    }
    div[data-testid="stMetric"] {
        background: #F1F8E9;
        border-radius: 15px;
        border: 1px solid #C8E6C9;
    }
    </style>
    <div class="top-header">
        <h2 style="color: white; margin: 0;">🎤 語音練習</h2>
    </div>
    <div class="subtitle">聽標準發音，然後錄下你的發音，系統會給你評分和建議</div>
    """, unsafe_allow_html=True)

# --- 2. 資料庫邏輯 ---
@st.cache_data
def load_db():
    return pd.read_csv("data/db.csv")

try:
    df = load_db()
except:
    st.error("找不到資料庫檔案")
    st.stop()

if 'current_task' not in st.session_state:
    st.session_state.current_task = df.sample(n=1).iloc[0]

# --- 3. 練習卡片主體 ---
task = st.session_state.current_task
st.markdown('<div class="practice-card">', unsafe_allow_html=True)
st.markdown('<span style="background:#E8F5E9; color:#2E7D32; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600;">難度：簡單</span>', unsafe_allow_html=True)

if os.path.exists(task['image_path']):
    st.image(task['image_path'], width=300)

st.markdown(f'<div class="word-display">{task["word"]}</div>', unsafe_allow_html=True)
st.audio(task['audio_path'])
st.markdown('</div>', unsafe_allow_html=True)

# --- 4. 錄音與精準評分演算法 ---
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    audio_data = mic_recorder(start_prompt="🎤 點擊開始錄音", stop_prompt="🛑 停止錄音", key='recorder')
    st.markdown('<p style="text-align:center; color:#888; font-size:13px;">點擊麥克風開始錄製發音</p>', unsafe_allow_html=True)

if audio_data:
    st.divider()
    with st.spinner("正在進行精密音頻分析..."):
        try:
            # 音軌轉換
            audio_bytes = audio_data['bytes']
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_seg.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # 載入音訊
            y_user, sr = librosa.load(wav_io, sr=16000)
            y_ref, _ = librosa.load(task['audio_path'], sr=16000)
            
            # 1. 自動切除靜音（避免空白段落稀釋分數）
            y_user, _ = librosa.effects.trim(y_user, top_db=20)
            y_ref, _ = librosa.effects.trim(y_ref, top_db=20)
            
            # 2. 提取多維特徵 (MFCC + Delta + Delta2)
            def extract_features(y, sr):
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                # 合併特徵並取平均值
                combined = np.concatenate([
                    np.mean(mfcc, axis=1), 
                    np.mean(delta, axis=1), 
                    np.mean(delta2, axis=1)
                ])
                return combined

            feat_ref = extract_features(y_ref, sr)
            feat_user = extract_features(y_user, sr)
            
            # 3. 計算餘弦相似度 (0~1)
            raw_sim = 1 - cosine(feat_ref, feat_user)
            
            # 4. 【重要】非線性評分縮放：拉開差距
            # 使用高次冪函數讓 0.9 以上才容易拿到高分，低於 0.8 的分數會大幅縮減
            if raw_sim < 0: raw_sim = 0
            adjusted_score = np.power(raw_sim, 5) * 100 
            
            # 修正邊界值
            final_score = int(min(max(adjusted_score, 0), 100))

            st.metric(label="🎯 發音精準度", value=f"{final_score} %")
            st.progress(final_score / 100)
            
            if final_score > 85:
                st.success("🎉 太棒了！你的發音與標準音契合度極高。")
            elif final_score > 60:
                st.info("👍 表現不錯，請嘗試注意發音細節後再挑戰！")
            else:
                st.warning("📖 差距較大，建議先多聽幾次標準發音喔。")
                
        except Exception as e:
            st.error(f"分析失敗，請重新嘗試。")

# --- 5. 底部功能 ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔄 隨機換一題", use_container_width=True):
    st.session_state.current_task = df.sample(n=1).iloc[0]
    st.rerun()