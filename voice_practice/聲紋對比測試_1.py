import streamlit as st
import pandas as pd
import librosa
import numpy as np
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io
import os


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

# --- 資料庫邏輯 ---
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


task = st.session_state.current_task
st.markdown('<div class="practice-card">', unsafe_allow_html=True)
st.markdown('<span style="background:#E8F5E9; color:#2E7D32; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600;">難度：簡單</span>', unsafe_allow_html=True)

if os.path.exists(task['image_path']):
    st.image(task['image_path'], width=300)

st.markdown(f'<div class="word-display">{task["word"]}</div>', unsafe_allow_html=True)
st.audio(task['audio_path'])
st.markdown('</div>', unsafe_allow_html=True)

# ---  錄音與 DTW 精準評分演算法 ---
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    audio_data = mic_recorder(start_prompt="🎤 點擊開始錄音", stop_prompt="🛑 停止錄音", key='recorder')
    st.markdown('<p style="text-align:center; color:#888; font-size:13px;">點擊麥克風開始錄製發音</p>', unsafe_allow_html=True)

if audio_data:
    st.divider()
    with st.spinner("正在進行時間序列對齊與精密分析..."):
        try:
            audio_bytes = audio_data['bytes']
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # --- 新增：音量檢查 ---
            if audio_seg.dBFS < -40:
                st.metric(label="🎯 發音精準度", value="0 %")
                st.progress(0)
                st.warning("⚠️ 系統偵測不到聲音，請靠近麥克風大聲練習喔！")
                st.stop()
            
            wav_io = io.BytesIO()
            audio_seg.export(wav_io, format="wav")
            wav_io.seek(0)
            
            y_user, sr = librosa.load(wav_io, sr=16000)
            y_ref, _ = librosa.load(task['audio_path'], sr=16000)
            
            # 1. 精準切除前後靜音
            y_user, _ = librosa.effects.trim(y_user, top_db=30)
            y_ref, _ = librosa.effects.trim(y_ref, top_db=30)
            
            # 2. 提取 MFCC (不取平均)
            mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
            mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr, n_mfcc=13)
            
            # 3. 使用 DTW 比對
            D, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_user, metric='cosine')
            avg_dist = D[-1, -1] / len(wp)
            
            # 4. 嚴格映射公式 
            raw_sim = max(0, 1 - (avg_dist * 1.5))
            
            # 5. 指數縮放
            final_score = int(np.power(raw_sim, 2.5) * 100)

            st.metric(label="🎯 發音精準度", value=f"{final_score} %")
            st.progress(final_score / 100)
            
            if final_score > 80:
                st.success("🎉 太棒了！發音非常標準。")
            elif final_score > 40:
                st.info("👍 還可以更好，再試試看！")
            else:
                st.warning("📖 差距較大，請多聽幾次標準發音。")
                
        except Exception as e:
            st.error(f"分析失敗，請重新嘗試。")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔄 隨機換一題", use_container_width=True):
    st.session_state.current_task = df.sample(n=1).iloc[0]
    st.rerun()