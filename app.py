import streamlit as st
from streamlit_mic_recorder import mic_recorder
import parselmouth
import whisper
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import os
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
            "세트 1: 어간 말 'ㄺ' 단순화 ('낡지' vs '낙지')",
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
            "세트 2: 어간 말 'ㄼ' 예외 단순화 ('밟도록' vs '밥도둑')",
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
    return whisper.load_model("base")

whisper_engine = get_whisper()

def extract_aligned_audio(audio_bytes, keyword, save_path, is_three_syllable=False):
    with open(save_path, "wb") as f:
        f.write(audio_bytes)
        
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

def compute_phonetic_metrics(sound):
    formants = sound.to_formant_burg(max_number_of_formants=5.0, maximum_formant=5500.0)
    spectrogram = sound.to_spectrogram(window_length=0.005)
    times = formants.ts()
    
    f2_vals = [formants.get_value_at_time(2, t) for t in times]
    f3_vals = [formants.get_value_at_time(3, t) for t in times]
    
    valid_f3 = [v for v in f3_vals if not np.isnan(v)]
    min_f3 = float(np.min(valid_f3)) if len(valid_f3) > 0 else 0.0
    mean_f3 = float(np.mean(valid_f3)) if len(valid_f3) > 0 else 0.0
    
    return {
        "sound": sound,
        "spectrogram": spectrogram,
        "times": times,
        "f2": f2_vals,
        "f3": f3_vals,
        "min_f3": min_f3,
        "mean_f3": mean_f3,
        "duration": float(sound.get_total_duration())
    }

def classify_pronunciation(data_target, data_control, set_info):
    f3_drop = data_control["min_f3"] - data_target["min_f3"]
    ratio = data_target["duration"] / (data_control["duration"] + 1e-6)
    
    if f3_drop > 180 and ratio >= 1.25:
        verdict = set_info["cand_hyper"]
        desc = "모음 [아] 말단의 F3 하강과 함께 조음 제스처 중첩으로 인한 지속시간 지연이 모두 뚜렷하여 두 자음 성분을 모두 의식한 이중조음형에 해당합니다."
    elif f3_drop > 220 and ratio < 1.25:
        verdict = set_info["cand_alt"]
        desc = "폐쇄음 조음이 약화되고 유음 [ㄹ] 성분이 지배적으로 실현되어 [ㄹ] 선택형 단순화 발음에 해당합니다."
    else:
        verdict = set_info["cand_std"]
        desc = "대조군과 F3 포먼트 궤적 및 폐쇄 구간 길이가 일치하며, 겹받침이 표준 규정에 맞게 단일 폐쇄음으로 깨끗이 단순화되었습니다."
        
    return verdict, f3_drop, ratio, desc

st.sidebar.title("🎛️ 메뉴 선택")
app_mode = st.sidebar.radio("모드 선택", ["학생 발음 실험 참여", "교수/연구자 관리자 모드"])

if app_mode == "학생 발음 실험 참여":
    st.title("🎙️ 국어음운론·형태론 현실 발음 대조 실습")
    st.markdown("자연스러운 문장 속에서 나의 겹받침 발음이 **어떤 음운 변이형에 가장 가까운지** 음향 분석을 통해 확인해 보세요.")

    with st.expander("👤 1단계: 연구 참가자 기본 정보 입력", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st_name = st.text_input("이름 (또는 닉네임)", placeholder="예: 국어과_김철수")
        with c2:
            st_gender = st.selectbox("성별", ["여성", "남성", "기타"])
        with c3:
            st_age = st.number_input("나이(만)", 17, 80, 22)
        with c4:
            st_region = st.selectbox("유년기 성장 지역", [
                "수도권(서울/경기/인천)", "경남(창원/마산/진주 등)", "부산", "대구", "경북",
                "충청도", "전라도", "강원도", "제주도", "기타/해외"
            ])

    conn = sqlite3.connect(DB_PATH)
    sets_df = pd.read_sql_query("SELECT * FROM stimulus_sets", conn)
    conn.close()

    set_options = sets_df["set_name"].tolist()
    chosen_set_name = st.selectbox("📌 2단계: 실습할 음운 대조 세트를 선택하세요", set_options)
    current_set = sets_df[sets_df["set_name"] == chosen_set_name].iloc[0]

    st.info(f"📘 **음운 규칙 해설:** {current_set['description']}")

    if "active_set" not in st.session_state or st.session_state.active_set != chosen_set_name:
        st.session_state.active_set = chosen_set_name
        st.session_state.rec_t = None
        st.session_state.rec_c = None

    st.markdown("### 🎙️ 3단계: 두 문장 발화 녹음")
    col_t, col_c = st.columns(2)
    with col_t:
        st.subheader(f"문장 A (표적: '{current_set['target_display']}')")
        st.warning(f"🗣️ **\"{current_set['sentence_target']}\"**")
        rec_t = mic_recorder(start_prompt="🔴 문장 A 녹음 시작", stop_prompt="⏹️ 녹음 완료", key="mic_t")
        if rec_t:
            st.session_state.rec_t = rec_t["bytes"]
        if st.session_state.rec_t:
            st.audio(st.session_state.rec_t, format="audio/wav")

    with col_c:
        st.subheader(f"문장 B (대조군: '{current_set['control_display']}')")
        st.info(f"🗣️ **\"{current_set['sentence_control']}\"**")
        rec_c = mic_recorder(start_prompt="🔴 문장 B 녹음 시작", stop_prompt="⏹️ 녹음 완료", key="mic_c")
        if rec_c:
            st.session_state.rec_c = rec_c["bytes"]
        if st.session_state.rec_c:
            st.audio(st.session_state.rec_c, format="audio/wav")

    if st.session_state.rec_t and st.session_state.rec_c:
        if not st_name.strip():
            st.error("⚠️ 상단 1단계에서 이름(또는 닉네임)을 입력해 주세요.")
        else:
            if st.button("🚀 내 발음 음향 분석 및 판정 결과 확인", use_container_width=True):
                with st.spinner("Whisper 단어 정렬 및 Praat 음향 지표 산출 중..."):
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path_t = os.path.join(AUDIO_DIR, f"{st_name}_{current_set['target_display']}_{stamp}.wav")
                    path_c = os.path.join(AUDIO_DIR, f"{st_name}_{current_set['control_display']}_{stamp}.wav")

                    is_three = ("밟" in current_set['target_keyword'])
                    snd_t, s_t, e_t = extract_aligned_audio(st.session_state.rec_t, current_set["target_keyword"], path_t, is_three)
                    snd_c, s_c, e_c = extract_aligned_audio(st.session_state.rec_c, current_set["control_keyword"], path_c, is_three)

                    prof_t = compute_phonetic_metrics(snd_t)
                    prof_c = compute_phonetic_metrics(snd_c)

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
                        chosen_set_name, current_set["target_display"], current_set["control_display"],
                        f3_drop, prof_t["duration"], prof_c["duration"], ratio,
                        verdict, path_t, path_c
                    ))
                    conn.commit()
                    conn.close()

                    st.markdown("---")
                    st.header(f"🎯 최종 판정 결과: 【 {verdict} 】")
                    st.write(desc)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("판정된 현실 발음형", verdict)
                    m2.metric("F3 포먼트 하강치 (대조군 대비)", f"{int(f3_drop)} Hz", delta=f"{int(-f3_drop)} Hz", delta_color="inverse")
                    m3.metric("조음 구간 지속시간 비율", f"{ratio:.2f} 배", delta=f"{(ratio-1.0)*100:+.1f}%")

                    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey="row")
                    
                    sg_t = prof_t["spectrogram"]
                    axes[0, 0].pcolormesh(sg_t.x_grid(), sg_t.y_grid(), 10 * np.log10(sg_t.values), cmap="viridis", shading="auto")
                    axes[0, 0].plot(prof_t["times"], prof_t["f3"], color="red", linewidth=2.5, label="F3 Track")
                    axes[0, 0].set_title(f"A. 표적 어절: '{current_set['target_display']}' (추출: {s_t:.2f}s~{e_t:.2f}s)")
                    axes[0, 0].set_ylim(0, 4500)
                    axes[0, 0].set_ylabel("Frequency (Hz)")
                    axes[0, 0].legend(loc="upper right")

                    sg_c = prof_c["spectrogram"]
                    axes[0, 1].pcolormesh(sg_c.x_grid(), sg_c.y_grid(), 10 * np.log10(sg_c.values), cmap="viridis", shading="auto")
                    axes[0, 1].plot(prof_c["times"], prof_c["f3"], color="red", linewidth=2.5, label="F3 Track")
                    axes[0, 1].set_title(f"B. 대조 어절: '{current_set['control_display']}' (추출: {s_c:.2f}s~{e_c:.2f}s)")
                    axes[0, 1].set_ylim(0, 4500)
                    axes[0, 1].legend(loc="upper right")

                    axes[1, 0].plot(snd_t.xs(), snd_t.values.T, color="#333333")
                    axes[1, 0].set_title(f"'{current_set['target_display']}' 음향 파형 (지속시간: {prof_t['duration']:.2f}s)")
                    axes[1, 0].set_xlabel("Time (s)")

                    axes[1, 1].plot(snd_c.xs(), snd_c.values.T, color="#005588")
                    axes[1, 1].set_title(f"'{current_set['control_display']}' 음향 파형 (지속시간: {prof_c['duration']:.2f}s)")
                    axes[1, 1].set_xlabel("Time (s)")

                    st.pyplot(fig)

elif app_mode == "교수/연구자 관리자 모드":
    st.title("🔒 국어음운론 연구 관리자 시스템")

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        st.subheader("관리자 로그인")
        with st.form("admin_login_form"):
            admin_id = st.text_input("관리자 아이디")
            admin_pw = st.text_input("비밀번호", type="password")
            submit_login = st.form_submit_button("로그인")

            if submit_login:
                if admin_id == "professor" and admin_pw == "linguist2026":
                    st.session_state.admin_logged_in = True
                    st.success("로그인 성공!")
                    st.rerun()
                else:
                    st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
        st.stop()

    col_out1, col_out2 = st.columns([8, 2])
    with col_out1:
        st.write("👋 **교수님 관리자 계정으로 접속 중입니다.**")
    with col_out2:
        if st.button("로그아웃"):
            st.session_state.admin_logged_in = False
            st.rerun()

    admin_tab1, admin_tab2, admin_tab3 = st.tabs([
        "📊 전체 발음 통계 분석", 
        "➕ 신규 자극문/분석 세트 추가", 
        "🎧 개별 음성 청취 및 CSV 다운로드"
    ])

    with admin_tab1:
        conn = sqlite3.connect(DB_PATH)
        df_res = pd.read_sql_query("SELECT * FROM participant_results ORDER BY id DESC", conn)
        conn.close()

        if df_res.empty:
            st.info("아직 수집된 학생 데이터가 없습니다.")
        else:
            st.subheader("📌 전체 발음 변이 실현율 요약")
            c1, c2, c3 = st.columns(3)
            c1.metric("총 분석 발화 건수", f"{len(df_res)} 건")
            c2.metric("참여 학생 수", f"{df_res['student_name'].nunique()} 명")
            std_ratio = (df_res['classified_label'].str.contains("표준")).mean() * 100
            c3.metric("표준 규칙 실현율", f"{std_ratio:.1f} %")

            st.markdown("---")
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                st.write("▼ **음운 변이형 분류 빈도**")
                st.bar_chart(df_res['classified_label'].value_counts())
            with col_ch2:
                st.write("▼ **출신 지역별 발음 변이 교차표**")
                ct = pd.crosstab(df_res['hometown'], df_res['classified_label'], margins=True)
                st.dataframe(ct, use_container_width=True)

    with admin_tab2:
        st.subheader("➕ 새로운 실험 자극문 등록")
        with st.form("new_stimulus_form"):
            new_set_name = st.text_input("실험 세트 제목", placeholder="예: 세트 3: 겹받침 'ㄺ'과 어미 '-ㄱ' 결합 ('맑게' vs '말게')")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                new_sent_t = st.text_input("표적 문장", placeholder="예: 하늘이 너무 맑게 개었어")
                new_key_t = st.text_input("표적 Whisper 검색어", placeholder="예: 맑")
                new_disp_t = st.text_input("표적 표시 어절", placeholder="예: 맑게")
            with col_s2:
                new_sent_c = st.text_input("대조 문장", placeholder="예: 국을 맑고 말게 끓여봐")
                new_key_c = st.text_input("대조 Whisper 검색어", placeholder="예: 말")
                new_disp_c = st.text_input("대조 표시 어절", placeholder="예: 말게")
            
            st.markdown("▼ **3분법 판정 레이블 설정**")
            cand_c1, cand_c2, cand_c3 = st.columns(3)
            with cand_c1:
                cand_1 = st.text_input("1. 표준 발음형", value="[말께] (표준: ㄹ단순화)")
            with cand_c2:
                cand_2 = st.text_input("2. 비표준 변이형", value="[막께] (비표준: ㄱ단순화)")
            with cand_c3:
                cand_3 = st.text_input("3. 이중조음/과도교정", value="[맑께] (이중조음형)")
            
            new_rule = st.text_area("음운 규칙 해설", placeholder="예: 표준어 규정 제11항 단서: 용언 어간 말 'ㄺ'은 'ㄱ' 앞에서 [ㄹ]로 발음")

            submit_new = st.form_submit_button("새 실험 세트 저장")
            if submit_new:
                if new_set_name and new_sent_t and new_sent_c:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    try:
                        c.execute('''
                            INSERT INTO stimulus_sets 
                            (set_name, sentence_target, sentence_control, target_keyword, control_keyword, 
                             target_display, control_display, cand_std, cand_alt, cand_hyper, description)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (new_set_name, new_sent_t, new_sent_c, new_key_t, new_key_c, 
                              new_disp_t, new_disp_c, cand_1, cand_2, cand_3, new_rule))
                        conn.commit()
                        st.success(f"'{new_set_name}' 세트가 성공적으로 등록되었습니다!")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("이미 존재하는 세트 제목입니다.")
                    finally:
                        conn.close()
                else:
                    st.warning("필수 항목을 모두 입력해 주세요.")

        st.markdown("---")
        st.subheader("📋 등록된 실험 세트 목록")
        conn = sqlite3.connect(DB_PATH)
        cur_sets = pd.read_sql_query("SELECT id, set_name, target_display, control_display, description FROM stimulus_sets", conn)
        conn.close()
        st.dataframe(cur_sets, use_container_width=True)

    with admin_tab3:
        st.subheader("📥 연구용 원시 데이터(CSV) 추출")
        conn = sqlite3.connect(DB_PATH)
        df_all = pd.read_sql_query("SELECT * FROM participant_results ORDER BY id DESC", conn)
        conn.close()

        if not df_all.empty:
            csv_bytes = df_all.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="💾 SPSS / R / Excel 호환 전체 원시 데이터 다운로드 (CSV)",
                data=csv_bytes,
                file_name=f"korean_phonetics_corpus_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("🎧 개별 학생 발화 음성 재청취")
            pick_id = st.selectbox(
                "확인할 데이터를 선택하세요:",
                df_all['id'].tolist(),
                format_func=lambda x: f"ID {x} | {df_all.loc[df_all['id']==x, 'student_name'].values[0]} ({df_all.loc[df_all['id']==x, 'hometown'].values[0]}) - {df_all.loc[df_all['id']==x, 'classified_label'].values[0]}"
            )
            row_data = df_all[df_all['id'] == pick_id].iloc[0]

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.caption(f"표적 어절 ('{row_data['target_display']}') 음성:")
                if os.path.exists(row_data['audio_path_target']):
                    st.audio(row_data['audio_path_target'])
            with col_p2:
                st.caption(f"대조 어절 ('{row_data['control_display']}') 음성:")
                if os.path.exists(row_data['audio_path_control']):
                    st.audio(row_data['audio_path_control'])