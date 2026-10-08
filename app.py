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
    (1, "세트 1: 어간 말 'ㄺ' 발음 조사 ('낡지' vs '낙지')", "신발이 너무 낡지 않았어?", "낙지가 너무 맛있지 않아?", "낡지", "낙지", "낡지", "낙지", "[낙찌] (표준: ㄱ단순화)", "[날찌] (비표준: ㄹ단순화)", "[낡찌] (과도교정: 이중조음)", "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"),
    (2, "세트 2: 어간 말 'ㄼ' 발음 조사 ('밟지' vs '밥지리')", "내 발 좀 밟지 마.", "전남에서는 벙어리를 밥지리라 한다.", "밟지", "밥지리", "밟지", "밥지", "[밥찌] (표준: ㅂ단순화)", "[발찌] (일반화 오류: ㄹ단순화)", "[밟찌] (과도교정: 이중조음)", "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"),
    (3, "세트 3: 어간 말 'ㄼ' 발음 조사 ('밟도록' vs '밥도둑')", "이 부분을 밟도록 해", "간장게장을 밥도둑이라고 해", "밟도록", "밥도둑", "밟도", "밥도", "[밥또록] (표준: ㅂ단순화)", "[발또록] (일반화 오류: ㄹ단순화)", "[밟또록] (과도교정: 이중조음)", "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음")
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS stimulus_sets (
        id INTEGER PRIMARY KEY, set_name TEXT UNIQUE, sentence_target TEXT, sentence_control TEXT,
        target_keyword TEXT, control_keyword TEXT, target_display TEXT, control_display TEXT,
        cand_std TEXT, cand_alt TEXT, cand_hyper TEXT, description TEXT)""")
        
    c.execute("""CREATE TABLE IF NOT EXISTS participant_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, student_name TEXT, gender TEXT, age INTEGER, hometown TEXT,
        set_name TEXT, target_display TEXT, control_display TEXT, f3_drop REAL, f3_drop_pct REAL, duration_target REAL, duration_control REAL,
        closure_ratio REAL, classified_label TEXT, perceived_label TEXT, expert_label TEXT, audio_path_target TEXT, audio_path_control TEXT)""")
        
    c.execute("""CREATE TABLE IF NOT EXISTS system_thresholds (id INTEGER PRIMARY KEY, th_f3_rel REAL, th_ratio REAL, updated_at TEXT)""")
    c.execute("INSERT OR IGNORE INTO system_thresholds (id, th_f3_rel, th_ratio, updated_at) VALUES (1, 1.8, 1.02, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    c.execute("PRAGMA table_info(participant_results)")
    cols = [r[1] for r in c.fetchall()]
    for col_name in ["perceived_label", "f3_drop_pct", "expert_label"]:
        if col_name not in cols:
            try: c.execute("ALTER TABLE participant_results ADD COLUMN " + col_name + " TEXT")
            except: pass

    for item in DEFAULT_SETS:
        c.execute("""INSERT OR REPLACE INTO stimulus_sets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", item)
    conn.commit()
    conn.close()

init_db()

def get_current_thresholds():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT th_f3_rel, th_ratio FROM system_thresholds WHERE id = 1")
    row = c.fetchone()
    conn.close()
    return (float(row[0]), float(row[1])) if row else (1.8, 1.02)

def auto_calibrate_thresholds():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT set_name, f3_drop_pct, closure_ratio, perceived_label, expert_label FROM participant_results WHERE f3_drop_pct IS NOT NULL", conn)
    conn.close()
    if len(df) < 4: return
    best_acc, best_tf, best_tr = -1.0, 1.8, 1.02
    for tf in [0.8, 1.2, 1.5, 1.8, 2.2]:
        for tr in [0.98, 1.01, 1.03, 1.05]:
            matches, vcnt = 0, 0
            for _, r in df.iterrows():
                ans = r['expert_label'] if pd.notnull(r['expert_label']) and r['expert_label'] != "" else r['perceived_label']
                if pd.isnull(ans) or ans == "": continue
                vcnt += 1
                is_b = ("밟" in str(r['set_name']))
                is_hyp = (r['closure_ratio'] >= tr or r['f3_drop_pct'] >= tf * 0.4) if is_b else ((r['f3_drop_pct'] >= tf and r['closure_ratio'] >= tr) or r['f3_drop_pct'] >= tf * 1.5)
                if is_hyp == ("낡" in str(ans) or "밟" in str(ans)): matches += 1
            if vcnt > 0 and (matches / vcnt) > best_acc:
                best_acc = matches / vcnt
                best_tf, best_tr = tf, tr
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE system_thresholds SET th_f3_rel = ?, th_ratio = ?, updated_at = ? WHERE id = 1", (best_tf, best_tr, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny")

def convert_and_save_audio(audio_bytes, out_path):
    temp = out_path + ".temp"
    with open(temp, "wb") as f: f.write(audio_bytes)
    subprocess.run(["ffmpeg", "-y", "-i", temp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp): os.remove(temp)

def extract_aligned_audio(audio_bytes, keyword, save_path, is_three=False):
    convert_and_save_audio(audio_bytes, save_path)
    res = get_whisper().transcribe(save_path, word_timestamps=True, language="ko")
    sound = parselmouth.Sound(save_path)
    total_d = sound.get_total_duration()

    t_start, t_end = None, None
    ckw = keyword.replace(" ", "").strip()
    skw = ckw[:2]

    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            wt = w["word"].replace(" ", "").strip()
            if (ckw in wt) or (skw in wt):
                t_start = float(w["start"])
                t_end = float(w["end"])
                break
        if t_start is not None: break

    valid = (t_start is not None and t_end is not None and 0.18 <= (t_end - t_start) <= 0.95)
    if not valid:
        if "낡" in keyword: t_start, t_end = total_d * 0.35, total_d * 0.35 + 0.42
        elif "낙" in keyword: t_start, t_end = total_d * 0.08, total_d * 0.08 + 0.42
        elif "밟도" in keyword: t_start, t_end = total_d * 0.32, total_d * 0.32 + 0.45
        elif "밟지" in keyword: t_start, t_end = total_d * 0.38, total_d * 0.38 + 0.42
        elif "밥지" in keyword: t_start, t_end = total_d * 0.42, total_d * 0.42 + 0.42
        else: t_start, t_end = total_d * 0.30, total_d * 0.30 + 0.45

    if is_three:
        t_end = t_start + (t_end - t_start) * 0.68

    p_s = max(0.0, t_start - 0.02)
    p_e = min(total_d, t_end + 0.02)
    part = sound.extract_part(from_time=p_s, to_time=p_e, preserve_times=False)
    return part, p_s, p_e

def compute_phonetic_metrics(sound, gender="남성"):
    max_f = 5000.0 if gender == "남성" else 5500.0
    formants = sound.to_formant_burg(max_number_of_formants=5.0, maximum_formant=max_f
