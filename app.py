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

SETS = [
    {
        "id": 1,
        "title": "과제 1: '낡지' vs '낙지'",
        "s_t": "신발이 너무 낡지 않았어?",
        "s_c": "낙지가 너무 맛있지 않아?",
        "kw_t": "낡지", "kw_c": "낙지",
        "disp_t": "낡지", "disp_c": "낙지",
        "cand": ["[낙찌] (표준: ㄱ단순화)", "[날찌] (비표준: ㄹ단순화)", "[낡찌] (과도교정: 이중조음)"],
        "rule": "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"
    },
    {
        "id": 2,
        "title": "과제 2: '밟지' vs '밥지리'",
        "s_t": "내 발 좀 밟지 마.",
        "s_c": "전남에서는 벙어리를 밥지리라 한다.",
        "kw_t": "밟지", "kw_c": "밥지리",
        "disp_t": "밟지", "disp_c": "밥지",
        "cand": ["[밥찌] (표준: ㅂ단순화)", "[발찌] (일반화 오류: ㄹ단순화)", "[밟찌] (과도교정: 이중조음)"],
        "rule": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"
    },
    {
        "id": 3,
        "title": "과제 3: '밟도록' vs '밥도둑'",
        "s_t": "이 부분을 밟도록 해",
        "s_c": "간장게장을 밥도둑이라고 해",
        "kw_t": "밟도록", "kw_c": "밥도둑",
        "disp_t": "밟도", "disp_c": "밥도",
        "cand": ["[밥또록] (표준: ㅂ단순화)", "[발또록] (일반화 오류: ㄹ단순화)", "[밟또록] (과도교정: 이중조음)"],
        "rule": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"
    }
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, name TEXT, gender TEXT, age INTEGER, region TEXT,
        task_id INTEGER, target_word TEXT, drop_pct REAL, ratio REAL,
        classified TEXT, perceived TEXT, expert TEXT, path_t TEXT, path_c TEXT)""")
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny")

def save_wav(audio_bytes, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(audio_bytes)
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(tmp):
        os.remove(tmp)

def slice_word(audio_bytes, kw, path, is_three=False):
    save_wav(audio_bytes, path)
    res = get_whisper().transcribe(path, word_timestamps=True, language="ko")
    snd = parselmouth.Sound(path)
    dur = float(snd.get_total_duration())

    t0, t1 = None, None
    clean_k = kw.replace(" ", "").strip()
    sub_k = clean_k[:2]

    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            txt = w["word"].replace(" ", "").strip()
            if (clean_k in txt) or (sub_k in txt):
                t0, t1 = float(w["start"]), float(w["end"])
                break
        if t0 is not None:
            break

    if (t0 is None) or (t1 is None) or not (0.18 <= (t1 - t0) <= 0.95):
        if "낡" in kw: t0, t1 = dur * 0.35, dur * 0.35 + 0.42
        elif "
