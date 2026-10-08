import streamlit as st
from streamlit_mic_recorder import mic_recorder
import parselmouth, whisper, numpy as np, matplotlib.pyplot as plt
import sqlite3, pandas as pd, os, subprocess
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="국어음운론 현실 발음 조사 실습실", layout="wide")

AUDIO_DIR = "recordings_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
DB_PATH = "phonetics_research.db"

SETS = [
    {"id": 1, "title": "과제 1: '낡지' vs '낙지'", "s_t": "신발이 너무 낡지 않았어?", "s_c": "낙지가 너무 맛있지 않아?", "kw_t": "낡지", "kw_c": "낙지", "disp_t": "낡지", "disp_c": "낙지", "cand": ["[낙찌] (표준: ㄱ단순화)", "[날찌] (비표준: ㄹ단순화)", "[낡찌] (과도교정: 이중조음)"], "rule": "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"},
    {"id": 2, "title": "과제 2: '밟지' vs '밥지리'", "s_t": "내 발 좀 밟지 마.", "s_c": "전남에서는 벙어리를 밥지리라 한다.", "kw_t": "밟지", "kw_c": "밥지리", "disp_t": "밟지", "disp_c": "밥지", "cand": ["[밥찌] (표준: ㅂ단순화)", "[발찌] (일반화 오류: ㄹ단순화)", "[밟찌] (과도교정: 이중조음)"], "rule": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"},
    {"id": 3, "title": "과제 3: '밟도록' vs '밥도둑'", "s_t": "이 부분을 밟도록 해", "s_c": "간장게장을 밥도둑이라고 해", "kw_t": "밟도록", "kw_c": "밥도둑", "disp_t": "밟도", "disp_c": "밥도", "cand": ["[밥또록] (표준: ㅂ단순화)", "[발또록] (일반화 오류: ㄹ단순화)", "[밟또록] (과도교정: 이중조음)"], "rule": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"}
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, name TEXT, gender TEXT, age INTEGER, region TEXT,
        task_id INTEGER, target_word TEXT, drop_pct REAL, ratio REAL, classified TEXT, perceived TEXT, expert TEXT, path_t TEXT, path_c TEXT)""")
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny")

def slice_word(audio_bytes, kw, path, is_3=False):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: f.write(audio_bytes)
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(tmp): os.remove(tmp)

    res = get_whisper().transcribe(path, word_timestamps=True, language="ko")
    snd = parselmouth.Sound(path)
    dur = float(snd.get_total_duration())
    t0, t1 = None, None
    ckw = kw.replace(" ", "").strip()[:2]

    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            if ckw in w["word"].replace(" ", "").strip():
                t0, t1 = float(w["start"]), float(w["end"])
                break
        if t0 is not None: break

    if (t0 is None) or not (0.18 <= (t1 - t0) <= 0.95):
        pos = 0.35 if "낡" in kw else (0.08 if "낙" in kw else (0.38 if "밟지" in kw else (0.42 if "밥지" in kw else 0.30)))
        t0, t1 = dur * pos, dur * pos + 0.42

    if is_3: t1 = t0 + (t1 - t0) * 0.68
    return snd.extract_part(from_time=max(0.0, t0 - 0.02), to_time=min(dur, t1 + 0.02), preserve_times=False)

def get_metrics(snd, gender="남성"):
    max_f = 5000.0 if gender == "남성" else 5500.0
    fmt = snd.to_formant_burg(max_number_of_formants=5.0, maximum_formant=max_f)
    sg = snd.to_spectrogram(window_length=0.005)
    dur = float(snd.get_total_duration())
    ts = fmt.ts()
    f3s = [fmt.get_value_at_time(3, t) for t in ts]

    off = [fmt.get_value_at_time(3, t) for t in ts if dur * 0.45 <= t <= dur * 0.82 and not np.isnan(fmt.get_value_at_time(3, t))]
    min_off = float(np.min(off)) if len(off) > 0 else 0.0
    mid = [fmt.get_value_at_time(3, t) for t in ts if dur * 0.50 <= t <= dur * 0.72 and not np.isnan(fmt.get_value_at_time(3, t))]
    avg_mid = float(np.mean(mid)) if len(mid) > 0 else min_off
    return {"snd": snd, "sg": sg, "ts": ts, "f3": f3s, "min_off": min_off, "avg_mid": avg_mid, "dur": dur}

def classify(m_t, m_c, task):
    drop_pct = ((m_c["min_off"] - m_t["min_off"]) / max(100.0, m_c["min_off"])) * 100.0
    ratio = m_t["dur"] / (m_c["dur"] + 1e-6)

    if "밟" in task["kw_t"]:
        m_drop = ((m_c["avg_mid"] - m_t["avg_mid"]) / max(100.0, m_c["avg_mid"])) * 100.0
        if ratio >= 1.01 or m_drop >= 0.8 or drop_pct >= 1.0:
            return task["cand"][2], drop_pct, ratio, "모음 후반부 F3 침하 및 조음 지속시간 연장 감지 ('이중조음형')"
        elif drop_pct >= 6.0 and ratio < 0.98:
            return task["cand"][1], drop_pct, ratio, "폐쇄음 약화 및 유음 성분 우세 ('[ㄹ] 선택형 단순화')"
        return task["cand"][0], drop_pct, ratio, "대조군과 음향 궤적 일치 (단일 양순 폐쇄음 규범 단순화)"
    else:
        if (drop_pct >= 1.8 and ratio >= 1.02) or drop_pct >= 2.7:
            return task["cand"][2], drop_pct, ratio, "모음 말단 F3 상대 하강 뚜렷 ('이중조음형')"
        elif drop_pct >= 4.5 and ratio < 1.02:
            return task["cand"][1], drop
