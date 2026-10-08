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

# Matplotlib 기본 설정 (리눅스 깨짐 방지)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="국어음운론 현실 발음 조사 실습실", page_icon="🎙️", layout="wide")

AUDIO_DIR = "recordings_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
DB_PATH = "phonetics_research.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS stimulus_sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT, set_name TEXT UNIQUE, sentence_target TEXT, sentence_control TEXT,
        target_keyword TEXT, control_keyword TEXT, target_display TEXT, control_display TEXT,
        cand_std TEXT, cand_alt TEXT, cand_hyper TEXT, description TEXT)""")
        
    c.execute("""CREATE TABLE IF NOT EXISTS participant_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, student_name TEXT, gender TEXT, age INTEGER, hometown TEXT,
        set_name TEXT, target_display TEXT, control_display TEXT, f3_drop REAL, f3_drop_pct REAL, duration_target REAL, duration_control REAL,
        closure_ratio REAL, classified_label TEXT, perceived_label TEXT, expert_label TEXT, audio_path_target TEXT, audio_path_control TEXT)""")
        
    c.execute("""CREATE TABLE IF NOT EXISTS system_thresholds (
        id INTEGER PRIMARY KEY,
        th_f3_rel REAL,
        th_ratio REAL,
        updated_at TEXT)""")
        
    c.execute("INSERT OR IGNORE INTO system_thresholds (id, th_f3_rel, th_ratio, updated_at) VALUES (1, 1.8, 1.02, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    c.execute("PRAGMA table_info(participant_results)")
    cols = [row[1] for row in c.fetchall()]
    if "perceived_label" not in cols:
        try: c.execute("ALTER TABLE participant_results ADD COLUMN perceived_label TEXT")
        except: pass
    if "f3_drop_pct" not in cols:
        try: c.execute("ALTER TABLE participant_results ADD COLUMN f3_drop_pct REAL")
        except: pass
    if "expert_label" not in cols:
        try: c.execute("ALTER TABLE participant_results ADD COLUMN expert_label TEXT")
        except: pass

    # 세트 1('낡지') -> 세트 2('밟지') -> 세트 3('밟도록') 순서로 정렬
    target_sets = [
        ("세트 1: 어간 말 'ㄺ' 발음 조사 ('낡지' vs '낙지')", "신발이 너무 낡지 않았어?", "낙지가 너무 맛있지 않아?", "낡지", "낙지", "낡지", "낙지", "[낙찌] (표준: ㄱ단순화)", "[날찌] (비표준: ㄹ단순화)", "[낡찌] (과도교정: 이중조음)", "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"),
        ("세트 2: 어간 말 'ㄼ' 발음 조사 ('밟지' vs '밥지리')", "내 발 좀 밟지 마.", "전남에서는 벙어리를 밥지리라 한다.", "밟지", "밥지리", "밟지", "밥지", "[밥찌] (표준: ㅂ단순화)", "[발찌] (일반화 오류: ㄹ단순화)", "[밟찌] (과도교정: 이중조음)", "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"),
        ("세트 3: 어간 말 'ㄼ' 발음 조사 ('밟도록' vs '밥도둑')", "이 부분을 밟도록 해", "간장게장을 밥도둑이라고 해", "밟도록", "밥도둑", "밟도", "밥도", "[밥또록] (표준: ㅂ단순화)", "[발또록] (일반화 오류: ㄹ단순화)", "[밟또록] (과도교정: 이중조음)", "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음")
    ]
    
    c.execute("DELETE FROM stimulus_sets")
    for item in target_sets:
        c.execute("""INSERT INTO stimulus_sets (set_name, sentence_target, sentence_control, target_keyword, control_keyword, target_display, control_display, cand_std, cand_alt, cand_hyper, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", item)
    conn.commit()
    conn.close()

init_db()

def get_current_thresholds():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT th_f3_rel, th_ratio FROM system_thresholds WHERE id = 1")
    row = c.fetchone()
    conn.close()
    if row:
        return float(row[0]), float(row[1])
    return 1.8, 1.02

def auto_calibrate_thresholds():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT set_name, f3_drop_pct, closure_ratio, perceived_label, expert_label FROM participant_results WHERE f3_drop_pct IS NOT NULL", conn)
    conn.close()
    
    if len(df) < 4:
        return
        
    best_acc = -1.0
    best_th_f3 = 1.8
    best_th_ratio = 1.02
    
    cand_f3 = [0.8, 1.2, 1.5, 1.8, 2.2, 2.8]
    cand_ratio = [0.98, 1.01, 1.03, 1.05]
    
    for tf in cand_f3:
        for tr in cand_ratio:
            matches = 0
            valid_cnt = 0
            for _, r in df.iterrows():
                target_ans = r['expert_label'] if pd.notnull(r['expert_label']) and r['expert_label'] != "" else r['perceived_label']
                if pd.isnull(target_ans) or target_ans == "":
                    continue
                valid_cnt += 1
                is_set_bilab = ("밟" in str(r['set_name']))
                if is_set_bilab:
                    is_hyper = (r['closure_ratio'] >= tr) or (r['f3_drop_pct'] >= tf * 0.4)
                else:
                    is_hyper = (r['f3_drop_pct'] >= tf and r['closure_ratio'] >= tr) or (r['f3_drop_pct'] >= tf * 1.5)
                    
                is_hyper_ans = ("낡" in str(target_ans) or "밟" in str(target_ans))
                if is_hyper == is_hyper_ans:
                    matches += 1
            if valid_cnt > 0:
                acc = matches / valid_cnt
                if acc > best_acc:
                    best_acc = acc
                    best_th_f3 = tf
                    best_th_ratio = tr
                
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE system_thresholds SET th_f3_rel = ?, th_ratio = ?, updated_at = ? WHERE id = 1", (best_th_f3, best_th_ratio, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny")

def convert_and_save_audio(audio_bytes, out_path):
    temp = out_path + ".temp"
    with open(temp, "wb") as f:
        f.write(audio_bytes)
    subprocess.run(["ffmpeg", "-y", "-i", temp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp):
        os.remove(temp)

def extract_aligned_audio(audio_bytes, keyword, save_path, is_three=False):
    convert_and_save_audio(audio_bytes, save_path)
    res = get_whisper().transcribe(save_path, word_timestamps=True, language="ko")
    sound = parselmouth.Sound(save_path)
    total_len = float(sound.get_total_duration())

    t_start, t_end = None, None
    clean_kw = keyword.replace(" ", "").strip()
    sub_kw = clean_kw[:2]

    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            w_text = w["word"].replace(" ", "").strip()
            if (clean_kw in w_text) or (sub_kw in w_text):
                t_start = float(w["start"])
