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
        
    cmd = [
        "ffmpeg", "-y", "-i", temp_input,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        output_wav_path
    ]
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
        from_time=max(0.0, t_start - 0.03),
        to_time=min(full_sound.get_total_duration(), t_end + 0.03),
        preserve_times=False
    )
    return part_sound, t_start, t_end

def compute_phonetic_metrics(sound, gender="남성"):
    max_formant = 5000.0 if gender == "남성" else 5500.0
    formants = sound.to_formant_burg(max_number_of_formants=5.0, maximum_formant=max_formant)
    spectrogram = sound.to_spectrogram(window_length=0.005)
    times = formants.ts()
    
    f2_vals = [formants.get_value_at_time(2, t) for t in times]
    f3_vals = [formants.get_value_at_time(3, t) for t in times]
    
    dur = float(sound.get_total_duration())
    
    t_offset_start = dur * 0.4
    t_offset_end = dur * 0.85
    offset_f3 = [
        formants.get_value_at_time(3, t) 
        for t in times 
        if (t >= t_offset_start and t <= t_offset_end and not np.isnan(formants.get_value_at_time(3, t)))
    ]
    
    min_offset_f3 = float(np.min(offset_f3)) if len(offset_f3) > 0 else 0.0
    mean_offset_f3 = float(np.mean(offset_f3)) if len(offset_f3) > 0 else 0.0
    
    return {
        "sound": sound,
        "spectrogram": spectrogram,
        "times": times,
        "f2": f2_vals,
        "f3": f3_vals,
        "min_offset_f3": min_offset_f3,
        "mean_offset_f3": mean_offset_f3,
        "duration": dur
    }

def classify_pronunciation(data_target, data_control, set_info):
    f3_th_hyper = 70.0
    ratio_th_hyper = 1.06
    f3_th_alt = 160.0

    f3_drop = data_control["min_offset_f3"] - data_target["min_offset_f3"]
    ratio = data_target["duration"] / (data_control["duration"] + 1e-6)
    
    if f3_drop >= f3_th_hyper and ratio >= ratio_th_hyper:
        verdict = set_info["cand_hyper"]
        desc = "모음 말단에서 F3 포먼트의 뚜렷한 하강과 함께 조음 제스처 중첩으로 인한 지속시간 연장이 관찰되어 [ㄹ]과 폐쇄음 성분이 함께 고려된 '이중조음형'으로 판정되었습니다."
    elif f3_drop >= f3_th_alt and ratio < ratio_th_hyper:
        verdict = set_info["cand_alt"]
        desc = "폐쇄음 성분이 약화되고 유음 [ㄹ] 성분이 지배적으로 실현되어 '[ㄹ] 선택형 단순화' 발음으로 판정되었습니다."
    elif f3_drop >= f3_th_hyper and ratio < ratio_th_hyper:
        verdict = set_info["cand_hyper"]
        desc = "발화 속도가 다소 빠르나 모음 말단 F3 궤적에서 유음화 조음 제스처의 잔여 효과가 포착되어 '이중조음형'으로 분류되었습니다."
    else:
        verdict = set_info["cand_std"]
        desc = "대조군과 F3 전이 궤적 및 지속시간 패턴이 일치하여, 겹받침이 규범에 맞게 단일 폐쇄음으로 깔끔하게 단순화되었습니다."
        
    return verdict, f3_drop, ratio, desc

def render_comparison_plots(snd_t, snd_c, prof_t, prof_c, target_word, control_word):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.5), sharey="row")
    
    sg_t = prof_t["spectrogram"]
    axes[0, 0].pcolormesh(sg_t.x_grid(), sg_t.y_grid(), 10 * np.log10(sg_t.values), cmap="viridis", shading="auto")
    axes[0, 0].plot(prof_t["times"], prof_t["f3"], color="red", linewidth=2.5, label="F3 Track")
    axes[0, 0].set_title(f"A. 표적 어절: '{target_word}' 스펙트로그램")
    axes[0, 0].set_ylim(0, 4500)
    axes[0, 0].set_ylabel("Frequency (Hz)")
    axes[0, 0].legend(loc="upper right")

    sg_c = prof_c["spectrogram"]
    axes[0, 1].pcolormesh(sg_c.x_grid(), sg_c.y_grid(), 10 * np.log10(sg_c.values), cmap="viridis", shading="auto")
    axes[0, 1].plot(prof_c["times"], prof_c["f3"], color="red", linewidth=2.5, label="F3 Track")
    axes[0, 1].set_title(f"B. 대조 어절: '{control_word}' 스펙트로그램")
    axes[0, 1].set_ylim(0, 4500)
    axes[0, 1].legend(loc="upper right")

    axes[1, 0].plot(snd_t.xs(), snd_t.values.T, color="#333333")
    axes[1, 0].set_title(f"'{target_word}' 파형 (지속시간: {prof_t['duration']:.2f}s)")
    axes[1, 0].set_xlabel("Time (s)")

    axes[1, 1].plot(snd_c.xs(), snd_c.values.T, color="#005588")
    axes[1, 1].set_title(f"'{control_word}' 파형 (지속시간: {prof_c['duration']:.2f}s)")
    axes[1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    return fig

st.sidebar.title("메뉴 선택")
app_mode = st.sidebar.radio("모드 선택", ["학생 발음 실험 참여", "교수/연구자 관리자 모드"])

if app_mode == "학생 발음 실험 참여":
    st.title("🎙️ 국어음운론·형태론 현실 발음 대조 실습")
    
    st.info(
        "📢 **실험 참여 안내**\n\n"
        "이 실험의 목적은 한국어 사용자의 현실 발음을 조사하는 것입니다. "
        "그러므로 표준 발음대로 발음하려고 시도하지 말고, 평소의 발음 습관대로 예문을 읽고 녹음해 주시기 바랍니다. "
        "녹음된 파일은 연구용으로만 사용되며 다른 용도로 이용되지 않습니다."
    )

    with st.expander("👤 1단계: 연구 참가자 기본 정보 입력", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st_name = st.text_input("이름 (또는 닉네임)", placeholder="예: 국어과_김철수")
        with c2:
            st_gender = st.selectbox("성별", ["여성", "남성", "기타"])
        with c3:
            st_age = st.number_input("나이(만)", min_value=5, max_value=90, value=22, step=1)
        with c4:
            st_region = st.selectbox("유년기 성장 지역", [
                "수도권(서울/경기/인천)", "경남(창원/마산/진주 등)", "부산", "대구", "경북",
                "충청도", "전라도", "강원도", "제주도", "기타/해외"
            ])

    conn = sqlite3.connect(DB_PATH)
    sets_df = pd.read_sql_query("SELECT * FROM stimulus_sets ORDER BY id ASC LIMIT 2", conn)
    conn.close()

    total_sets = len(sets_df)
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    step_idx = st.session_state.current_step

    if step_idx < total_sets:
        current_set = sets_df.iloc[step_idx]

        progress_val = (step_idx) / total_sets
        st.progress(progress_val, text=f"전체 실험 진행 상황: {step_idx + 1} / {total_sets} 단계 진행 중")
        st.subheader(f"과제 {step_idx + 1}")

        step_rec_t_key = f"rec_t_{current_set['id']}"
        step_rec_c_key = f"rec_c_{current_set['id']}"
        if step_rec_t_key not in st.session_state:
            st.session_state[step_rec_t_key] = None
        if step_rec_c_key not in st.session_state:
            st.session_state[step_rec_c_key] = None

        st.markdown("### 🎙️ 발화 녹음")
        st.caption("※ 평소 말하는 속도와 억양으로 자연스럽게 소리 내어 읽어주세요.")
        col_t, col_c = st.columns(2)
        
        target_txt = current_set['sentence_target']
        control_txt = current_set['sentence_control']
        target_word = current_set['target_display']
        control_word = current_set['control_display']

        with col_t:
            st.markdown("**문장 1**")
            st.warning(f"🗣️ {target_txt}")
            rec_t = mic_recorder(start_prompt="🔴 문장 1 녹음 시작", stop_prompt="⏹️ 녹음 완료", key=f"mic_t_{step_idx}")
            if rec_t:
                st.session_state[step_rec_t_key] = rec_t["bytes"]
            if st.session_state[step_rec_t_key]:
                st.audio(st.session_state[step_rec_t_key], format="audio/wav")

        with col_c:
            st.markdown("**문장 2**")
            st.info(f"🗣️ {control_txt}")
            rec_c = mic_recorder(start_prompt="🔴 문장 2 녹음 시작", stop_prompt="⏹️ 녹음 완료", key=f"mic_c_{step_idx}")
            if rec_c:
                st.session_state[step_rec_c_key] = rec_c["bytes"]
            if st.session_state[step_rec_c_key]:
                st.audio(st.session_state[step_rec_c_key], format="audio/wav")

        if st.session_state[step_rec_t_key] and st.session_state[step_rec_c_key]:
            if not st_name.strip():
                st.error("⚠️ 상단 1단계에서 이름(또는 닉네임)을 입력해 주세요.")
            else:
                if st.button("🚀 녹음 완료 및 음향 분석 실행", key=f"btn_eval_{step_idx}", use_container_width=True):
                    with st.spinner("단어 정렬 및 음향 지표 산출 중..."):
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path_t = os.path.join(AUDIO_DIR, f"{st_name}_{target_word}_{stamp}.wav")
                        path_c = os.path.join(AUDIO_DIR, f"{st_name}_{control_word}_{stamp}.wav")

                        is_three = ("밟" in current_set['target_keyword'])
                        snd_t, s_t, e_t = extract_aligned_audio(st.session_state[step_rec_t_key], current_set["target_keyword"], path_t, is_three)
                        snd_c, s_c, e_c = extract_aligned_audio(st.session_state[step_rec_c_key], current_set["control_keyword"], path_c, is_three)

                        prof_t = compute_phonetic_metrics(snd_t, gender=st_gender)
                        prof_c = compute_phonetic_metrics(snd_c, gender=st_gender)

                        verdict, f3_drop, ratio, desc = classify_pronunciation(prof_t, prof_c, current_set)

                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute('''
                            INSERT INTO participant_results (
                                created_at, student_name, gender, age, hometown,
                                set_name, target_display, control_display,
                                f3_drop, duration_target, duration_control, closure_ratio,
                                classified_label, audio_path_target, audio_path_control
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            st_name.strip(), st_gender, st_age, st_region,
                            current_set['set_name'], target_word, control_word,
                            f3_drop, prof_t["duration"], prof_c["duration"], ratio,
                            verdict, path_t, path_c
                        ))
                        conn.commit()
                        conn.close()

                        st.session_state[f"evaluated_{step_idx}"] = {
                            "verdict": verdict,
                            "f3_drop": f3_drop,
                            "ratio": ratio,
                            "desc": desc,
                            "cand_std": current_set['cand_std'],
                            "cand_alt": current_set['cand_alt'],
                            "cand_hyper": current_set['cand_hyper'],
                            "rule_desc": current_set['description'],
                            "prof_t": prof_t,
                            "prof_c": prof_c,
                            "snd_t": snd_t,
                            "snd_c": snd_c,
                            "s_t": s_t, "e_t": e_t,
                            "s_c": s_c, "e_c": e_c
                        }

        if f"evaluated_{step_idx}" in st.session_state:
            ev = st.session_state[f"evaluated_{step_idx}"]
            st.markdown("---")
