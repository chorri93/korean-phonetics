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

    initial = [
        ("세트 1: 어간 말 'ㄺ' 발음 조사 ('낡지' vs '낙지')", "신발이 너무 낡지 않았어?", "낙지가 너무 맛있지 않아?", "낡지", "낙지", "낡지", "낙지", "[낙찌] (표준: ㄱ단순화)", "[날찌] (비표준: ㄹ단순화)", "[낡찌] (과도교정: 이중조음)", "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"),
        ("세트 2: 어간 말 'ㄼ' 발음 조사 ('밟도록' vs '밥도둑')", "이 부분을 밟도록 해", "간장게장을 밥도둑이라고 해", "밟도록", "밥도둑", "밟도", "밥도", "[밥또록] (표준: ㅂ단순화)", "[발또록] (일반화 오류: ㄹ단순화)", "[밟또록] (과도교정: 이중조음)", "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음")
    ]
    for item in initial:
        c.execute("""INSERT OR IGNORE INTO stimulus_sets (set_name, sentence_target, sentence_control, target_keyword, control_keyword, target_display, control_display, cand_std, cand_alt, cand_hyper, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", item)
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
                is_set2 = ("밟" in str(r['set_name']))
                if is_set2:
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
                t_end = float(w["end"])
                break
        if t_start is not None:
            break

    is_valid_range = False
    if t_start is not None and t_end is not None:
        seg_dur = t_end - t_start
        if 0.18 <= seg_dur <= 0.95:
            is_valid_range = True

    if not is_valid_range:
        if "낡" in keyword:
            t_start = max(0.2, total_len * 0.35)
            t_end = min(total_len - 0.2, t_start + 0.42)
        elif "낙" in keyword:
            t_start = max(0.1, total_len * 0.08)
            t_end = min(total_len - 0.2, t_start + 0.42)
        elif "밟" in keyword:
            t_start = max(0.2, total_len * 0.32)
            t_end = min(total_len - 0.2, t_start + 0.45)
        else:
            t_start = max(0.2, total_len * 0.30)
            t_end = min(total_len - 0.2, t_start + 0.45)

    if is_three:
        diff = t_end - t_start
        t_end = t_start + diff * 0.68

    p_start = max(0.0, t_start - 0.02)
    p_end = min(total_len, t_end + 0.02)
    part = sound.extract_part(from_time=p_start, to_time=p_end, preserve_times=False)
    return part, p_start, p_end

def compute_phonetic_metrics(sound, gender="남성"):
    max_f = 5000.0 if gender == "남성" else 5500.0
    formants = sound.to_formant_burg(max_number_of_formants=5.0, maximum_formant=max_f)
    spectrogram = sound.to_spectrogram(window_length=0.005)
    times = formants.ts()
    
    f3_vals = []
    for t in times:
        f3_vals.append(formants.get_value_at_time(3, t))
        
    dur = float(sound.get_total_duration())
    
    t1, t2 = dur * 0.45, dur * 0.82
    offset_f3 = []
    for t in times:
        if t1 <= t <= t2:
            val = formants.get_value_at_time(3, t)
            if not np.isnan(val):
                offset_f3.append(val)
    min_off = float(np.min(offset_f3)) if len(offset_f3) > 0 else 0.0

    m1, m2 = dur * 0.50, dur * 0.72
    mid_f3 = []
    for t in times:
        if m1 <= t <= m2:
            val = formants.get_value_at_time(3, t)
            if not np.isnan(val):
                mid_f3.append(val)
    avg_mid_f3 = float(np.mean(mid_f3)) if len(mid_f3) > 0 else min_off

    return {"sound": sound, "spectrogram": spectrogram, "times": times, "f3": f3_vals, "min_off": min_off, "avg_mid_f3": avg_mid_f3, "duration": dur}

def classify_pronunciation_adaptive(d_t, d_c, s_info):
    th_f3_rel, th_ratio = get_current_thresholds()
    
    drop_hz = d_c["min_off"] - d_t["min_off"]
    base_f3 = max(100.0, d_c["min_off"])
    drop_pct = (drop_hz / base_f3) * 100.0
    
    denom = d_c["duration"] + 0.000001
    ratio = d_t["duration"] / denom

    is_bilabial_set = ("밟" in s_info["target_keyword"])

    if is_bilabial_set:
        mid_drop_hz = d_c["avg_mid_f3"] - d_t["avg_mid_f3"]
        mid_drop_pct = (mid_drop_hz / max(100.0, d_c["avg_mid_f3"])) * 100.0
        is_hyper = (ratio >= 1.01) or (mid_drop_pct >= 0.8) or (drop_pct >= 1.0)
        
        if is_hyper:
            verdict = s_info["cand_hyper"]
            desc = f"모음 후반부 F3 침하({mid_drop_pct:.1f}%) 및 조음 지속시간({ratio:.2f}배) 분석을 통해 '이중조음형([밟또록])'으로 판정되었습니다."
        elif drop_pct >= 6.0 and ratio < 0.98:
            verdict = s_info["cand_alt"]
            desc = "폐쇄음이 완전히 약화되고 유음 성분이 지배적이어서 '[ㄹ] 선택형 단순화([발또록])'로 판정되었습니다."
        else:
            verdict = s_info["cand_std"]
            desc = f"대조군('밥도둑')과 포먼트 궤적이 일치하여 단일 양순 폐쇄음 규범 단순화([밥또록])로 판정되었습니다."
    else:
        if (drop_pct >= th_f3_rel and ratio >= th_ratio) or (drop_pct >= th_f3_rel * 1.5):
            verdict = s_info["cand_hyper"]
            desc = f"모음 말단 F3 상대 하강율이 {drop_pct:.1f}%(기준: {th_f3_rel:.1f}%)로 유음화 조음 제스처가 잔류하여 '이중조음형([낡찌])'으로 판정되었습니다."
        elif drop_pct >= th_f3_rel * 2.5 and ratio < th_ratio:
            verdict = s_info["cand_alt"]
            desc = "폐쇄음이 약화되고 유음 성분이 지배적이어서 '[ㄹ] 선택형 단순화([날찌])'로 판정되었습니다."
        else:
            verdict = s_info["cand_std"]
            desc = f"대조군 대비 F3 변동폭({drop_pct:.1f}%)이 정상 범위 내에 있어 단일 연구개 폐쇄음 규범 단순화([낙찌])로 판정되었습니다."
        
    return verdict, drop_hz, drop_pct, ratio, desc

def plot_phonetics(snd_t, snd_c, pf_t, pf_c, w_t, w_c):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.5), sharey="row")
    sg_t, sg_c = pf_t["spectrogram"], pf_c["spectrogram"]
    
    axes[0, 0].pcolormesh(sg_t.x_grid(), sg_t.y_grid(), 10 * np.log10(sg_t.values), cmap="viridis", shading="auto")
    axes[0, 0].plot(pf_t["times"], pf_t["f3"], color="red", linewidth=2.5, label="F3 Track")
    axes[0, 0].set_title(f"Target: '{w_t}' Spectrogram", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(0, 4500)
    axes[0, 0].set_ylabel("Frequency (Hz)")
    axes[0, 0].legend(loc="upper right")
    
    axes[0, 1].pcolormesh(sg_c.x_grid(), sg_c.y_grid(), 10 * np.log10(sg_c.values), cmap="viridis", shading="auto")
    axes[0, 1].plot(pf_c["times"], pf_c["f3"], color="red", linewidth=2.5, label="F3 Track")
    axes[0, 1].set_title(f"Control: '{w_c}' Spectrogram", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 4500)
    axes[0, 1].legend(loc="upper right")
    
    axes[1, 0].plot(snd_t.xs(), snd_t.values.T, color="#333")
    axes[1, 0].set_title(f"'{w_t}' Waveform ({pf_t['duration']:.2f}s)", fontsize=11)
    axes[1, 0].set_xlabel("Time (s)")
    
    axes[1, 1].plot(snd_c.xs(), snd_c.values.T, color="#005588")
    axes[1, 1].set_title(f"'{w_c}' Waveform ({pf_c['duration']:.2f}s)", fontsize=11)
    axes[1, 1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    return fig

st.sidebar.title("메뉴 선택")
app_mode = st.sidebar.radio("모드 선택", ["학생 발음 실험 참여", "교수/연구자 관리자 모드"])

# ==========================================================
# 1. 학생 발음 실험 참여 (개별 학생용 UI: 스펙트로그램 분석 결과 제시)
# ==========================================================
if app_mode == "학생 발음 실험 참여":
    st.title("🎙️ 국어음운론 현실 발음 대조 실습")
    st.info("📢 **실험 참여 안내**\n\n이 실험의 목적은 한국어 사용자의 현실 발음을 조사하는 것입니다. 그러므로 표준 발음대로 발음하려고 시도하지 말고, 평소의 발음 습관대로 예문을 읽고 녹음해 주시기 바랍니다. 녹음된 파일은 연구용으로만 사용되며 다른 용도로 이용되지 않습니다.")

    with st.expander("👤 1단계: 연구 참가자 기본 정보 입력", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: st_name = st.text_input("이름 (또는 닉네임)")
        with c2: st_gender = st.selectbox("성별", ["여성", "남성", "기타"])
        with c3: st_age = st.number_input("나이(만)", 5, 90, 22, 1)
        with c4: st_region = st.selectbox("유년기 성장 지역", ["수도권(서울/경기/인천)", "경남(창원/마산/진주 등)", "부산", "대구", "경북", "충청도", "전라도", "강원도", "제주도", "기타/해외"])

    conn = sqlite3.connect(DB_PATH)
    sets_df = pd.read_sql_query("SELECT * FROM stimulus_sets ORDER BY id ASC LIMIT 2", conn)
    conn.close()

    total_sets = len(sets_df)
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    step_idx = st.session_state.current_step

    if step_idx < total_sets:
        cur_set = sets_df.iloc[step_idx]
        st.progress(step_idx / total_sets, text=f"전체 실험 진행 상황: {step_idx + 1} / {total_sets} 단계")
        st.subheader(f"과제 {step_idx + 1}")

        k_t, k_c = f"rec_t_{cur_set['id']}", f"rec_c_{cur_set['id']}"
        if k_t not in st.session_state: st.session_state[k_t] = None
        if k_c not in st.session_state: st.session_state[k_c] = None

        st.caption("※ 평소 말하는 속도와 억양으로 자연스럽게 소리 내어 읽어주세요.")
        col_t, col_c = st.columns(2)
        with col_t:
            st.markdown("**문장 1**")
            st.warning(f"🗣️ {cur_set['sentence_target']}")
            r_t = mic_recorder(start_prompt="🔴 문장 1 녹음 시작", stop_prompt="⏹️ 녹음 완료", key=f"mic_t_{step_idx}")
            if r_t: st.session_state[k_t] = r_t["bytes"]
            if st.session_state[k_t]: st.audio(st.session_state[k_t], format="audio/wav")
        with col_c:
            st.markdown("**문장 2**")
            st.info(f"🗣️ {cur_set['sentence_control']}")
            r_c = mic_recorder(start_prompt="🔴 문장 2 녹음 시작", stop_prompt="⏹️ 녹음 완료", key=f"mic_c_{step_idx}")
            if r_c: st.session_state[k_c] = r_c["bytes"]
            if st.session_state[k_c]: st.audio(st.session_state[k_c], format="audio/wav")

        if st.session_state[k_t] and st.session_state[k_c]:
            st.markdown("---")
            t_disp = cur_set['target_display']
            p1 = cur_set['cand_std'].split(" ")[0]
            p2 = cur_set['cand_alt'].split(" ")[0]
            p3 = cur_set['cand_hyper'].split(" ")[0]
            perceived_options = [p1, p2, p3]

            st.markdown(f"#### 🤔 방금 읽으신 '{t_disp}'를 어떻게 발음했다고 느끼셨나요?")
            user_perceived = st.radio("본인이 실제로 소리 낸 발음을 하나 선택해 주세요:", perceived_options, key=f"perceive_{step_idx}", horizontal=True)

            if not st_name.strip():
                st.error("⚠️ 상단 1단계에서 이름을 입력해 주세요.")
            elif st.button("🚀 녹음 완료 및 음향 분석 실행", key=f"btn_eval_{step_idx}", use_container_width=True):
                with st.spinner("표적 어절 정밀 추출 및 음향 지표 산출 중..."):
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pt = os.path.join(AUDIO_DIR, f"{st_name}_{cur_set['target_display']}_{stamp}.wav")
                    pc = os.path.join(AUDIO_DIR, f"{st_name}_{cur_set['control_display']}_{stamp}.wav")
                    is_3 = ("밟" in cur_set['target_keyword'])

                    snd_t, st_t, ed_t = extract_aligned_audio(st.session_state[k_t], cur_set["target_keyword"], pt, is_3)
                    snd_c, st_c, ed_c = extract_aligned_audio(st.session_state[k_c], cur_set["control_keyword"], pc, is_3)
                    pf_t = compute_phonetic_metrics(snd_t, st_gender)
                    pf_c = compute_phonetic_metrics(snd_c, st_gender)
                    
                    verdict, drop_hz, drop_pct, ratio, desc = classify_pronunciation_adaptive(pf_t, pf_c, cur_set)

                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""INSERT INTO participant_results (created_at, student_name, gender, age, hometown, set_name, target_display, control_display, f3_drop, f3_drop_pct, duration_target, duration_control, closure_ratio, classified_label, perceived_label, expert_label, audio_path_target, audio_path_control) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), st_name.strip(), st_gender, st_age, st_region, cur_set['set_name'], cur_set['target_display'], cur_set['control_display'], drop_hz, drop_pct, pf_t["duration"], pf_c["duration"], ratio, verdict, user_perceived, None, pt, pc))
                    conn.commit()
                    conn.close()

                    auto_calibrate_thresholds()

                    st.session_state[f"eval_{step_idx}"] = {
                        "verdict": verdict, "drop_hz": drop_hz, "drop_pct": drop_pct, "ratio": ratio, "desc": desc,
                        "perceived": user_perceived, "snd_t": snd_t, "snd_c": snd_c,
                        "pf_t": pf_t, "pf_c": pf_c, "set": cur_set
                    }

        # 학생 결과 리포트 (스펙트로그램 분석 결과 중심)
        if f"eval_{step_idx}" in st.session_state:
            ev = st.session_state[f"eval_{step_idx}"]
            st.markdown("---")
            st.subheader("🎯 스펙트로그램 음향 분석 결과")
            cands = [("표준 발음형", ev['set']['cand_std']), ("방언/비표준 변이형", ev['set']['cand_alt']), ("이중조음/과도교정", ev['set']['cand_hyper'])]
            cols = st.columns(3)
            for i, (lab, val) in enumerate(cands):
                with cols[i]:
                    if val == ev['verdict']:
                        st.markdown(f"""<div style="border: 3px solid #FF4B4B; background: rgba(255,75,75,0.08); padding: 15px; border-radius: 10px; text-align: center;"><b style="color: #
