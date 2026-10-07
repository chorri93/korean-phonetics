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

st.set_page_config(
    page_title="국어음운론 현실 발음 조사 실습실",
    page_icon="🎙️",
    layout="wide"
)

AUDIO_DIR = "recordings_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
DB_PATH = "phonetics_research.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS stimulus_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            set_name TEXT UNIQUE,
            sentence_target TEXT,
            sentence_control TEXT,
            target_keyword TEXT,
            control_keyword TEXT,
            target_display TEXT,
            control_display TEXT,
            cand_std TEXT,
            cand_alt TEXT,
            cand_hyper TEXT,
            description TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS participant_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            student_name TEXT,
            gender TEXT,
            age INTEGER,
            hometown TEXT,
            set_name TEXT,
            target_display TEXT,
            control_display TEXT,
            f3_drop REAL,
            duration_target REAL,
            duration_control REAL,
            closure_ratio REAL,
            classified_label TEXT,
            audio_path_target TEXT,
            audio_path_control TEXT
        )
    ''')
    
    initial_sets = [
        (
            "세트 1: 어간 말 'ㄺ' 발음 조사 ('낡지' vs '낙지')",
            "신발이 너무 낡지 않았어?",
            "낙지가 너무 맛있지 않아?",
            "낡", "낙",
            "낡지", "낙지",
            "[낙찌] (표준: ㄱ단순화)",
            "[날찌] (비표준: ㄹ단순화)",
            "[낡찌] (과도교정: 이중조음)",
            "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"
        ),
        (
            "세트 2: 어간 말 'ㄼ' 발음 조사 ('밟도록' vs '밥도둑')",
            "이 부분을 밟도록 해",
            "간장게장을 밥도둑이라고 해",
            "밟", "밥",
            "밟도", "밥도",
            "[밥또록] (표준: ㅂ단순화)",
            "[발또록] (일반화 오류: ㄹ단순화)",
            "[밟또록] (과도교정: 이중조음)",
            "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"
        )
    ]
    for item in initial_sets:
        c.execute('''
            INSERT OR IGNORE INTO stimulus_sets 
            (set_name, sentence_target, sentence_control, target_keyword, control_keyword, 
             target_display, control_display, cand_std, cand_alt, cand_hyper, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', item)
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny")

def convert_and_save_audio(audio_bytes, output_wav_path):
    temp_input = output_wav_path + ".temp"
    with open(temp_input, "wb") as f:
        f.write(audio_bytes)
    cmd = ["ffmpeg", "-y", "-i", temp_input, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output_wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_input):
        os.remove(temp_input)

def extract_aligned_audio(audio_bytes, keyword, save_path, is_three_syllable=False):
    convert_and_save_audio(audio_bytes, save_path)
    whisper_engine = get_whisper()
    res = whisper_engine.transcribe(save_path, word_timestamps=True, language="ko")
    
    t_start, t_end = None, None
    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            clean = w["word"].replace(" ", "").strip()
            if keyword in clean:
                t_start = w["start"]
                t_end = w["end"]
                break
        if t_start is not None:
            break
            
    full_sound = parselmouth.Sound(save_path)
    if t_start is None or t_end is None:
        return full_sound, 0.0, full_sound.get_total_duration()
        
    if is_three_syllable:
        span = t_end - t_start
        t_end = t_start + (span * 0.68)
        
    part_sound = full_sound.extract_part(
        from_time=max(0.0, t_start - 0
