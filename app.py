import streamlit as st
from streamlit_mic_recorder import mic_recorder
import parselmouth
import whisper
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import os
import subprocess
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="국어음운론 현실 발음 조사 실습실", page_icon="🎙️", layout="wide")

AUDIO_DIR = "recordings_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
DB_PATH = "phonetics_research.db"

DEFAULT_SETS = [
    {
        "id": 1,
        "set_name": "세트 1: 어간 말 'ㄺ' 발음 조사 ('낡지' vs '낙지')",
        "sentence_target": "신발이 너무 낡지 않았어?",
        "sentence_control": "낙지가 너무 맛있지 않아?",
        "target_keyword": "낡지",
        "control_keyword": "낙지",
        "target_display": "낡지",
        "control_display": "낙지",
        "cand_std": "[낙찌] (표준: ㄱ단순화)",
        "cand_alt": "[날찌] (비표준: ㄹ단순화)",
        "cand_hyper": "[낡찌] (과도교정: 이중조음)",
        "description": "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"
    },
    {
        "id": 2,
        "set_name": "세트 2: 어간 말 'ㄼ' 발음 조사 ('밟지' vs '밥지리')",
        "sentence_target": "내 발 좀 밟지 마.",
        "sentence_control": "전남에서는 벙어리를 밥지리라 한다.",
        "target_keyword": "밟지",
        "control_keyword": "밥지리",
        "target_display": "밟지",
        "control_display": "밥지",
        "cand_std": "[밥찌] (표준: ㅂ단순화)",
        "cand_alt": "[발찌] (일반화 오류: ㄹ단순화)",
        "cand_hyper": "[밟찌] (과도교정: 이중조음)",
        "description": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"
    },
    {
        "id": 3,
        "set_name": "세트 3: 어간 말 'ㄼ' 발음 조사 ('밟도록' vs '밥도둑')",
        "sentence_target": "이 부분을 밟도록 해",
        "sentence_control": "간장게장을 밥도둑이라고 해",
        "target_keyword": "밟도록",
        "control_keyword": "밥도둑",
        "target_display": "밟도",
        "control_display": "밥도",
        "cand_std": "[밥또록] (표준: ㅂ단순화)",
        "cand_alt": "[발또록] (일반화 오류: ㄹ단순화)",
        "cand_hyper": "[밟또록] (과도교정: 이중조음)",
        "description": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"
    }
]

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS participant_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, student_name TEXT, gender TEXT, age INTEGER, hometown TEXT,
            set_name TEXT, target_display TEXT, control_display TEXT, f3_drop REAL, f3_drop_pct REAL, duration_target REAL, duration_control REAL,
            closure_ratio REAL, classified_label TEXT, perceived_label TEXT, expert_label TEXT, audio_path_target TEXT, audio_path_control TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS system_thresholds (id INTEGER PRIMARY KEY, th_f3_rel REAL, th_ratio REAL, updated_at TEXT)""")
        c.execute("INSERT OR IGNORE INTO system_thresholds (id, th_f3_rel, th_ratio, updated_at) VALUES (1, 1.8, 1.02, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
        conn.commit()
        conn.close()
    except Exception:
        pass

init_db()

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny")

def convert_and_save_audio(audio_bytes, out_path):
    temp = out_path + ".temp"
    with open(temp, "wb") as f:
