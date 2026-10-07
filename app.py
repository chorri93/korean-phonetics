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
        set_name TEXT, target_display TEXT, control_display TEXT, f3_drop REAL, duration_target REAL, duration_control REAL,
        closure_ratio REAL, classified_label TEXT, audio_path_target TEXT, audio_path_control TEXT)""")
    initial = [
        ("세트 1: 어간 말 'ㄺ' 발음 조사 ('낡지' vs '낙지')", "신발이 너무 낡지 않았어?", "낙지가 너무 맛있지 않아?", "낡", "낙", "낡지", "낙지", "[낙찌] (표준: ㄱ단순화)", "[날찌] (비표준: ㄹ단순화)", "[낡찌] (과도교정: 이중조음)", "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"),
        ("세트 2: 어간 말 'ㄼ' 발음 조사 ('밟도록' vs '밥도둑')", "이 부분을 밟도록 해", "간장게장을 밥도둑이라고 해", "밟", "밥", "밟도", "밥도", "[밥또록] (표준: ㅂ단순화)", "[발또록] (일반화 오류: ㄹ단순화)", "[밟또록] (과도교정: 이중조음)", "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음")
    ]
    for item in initial:
        c.execute("""INSERT OR IGNORE INTO stimulus_sets (set_name, sentence_target, sentence_control, target_keyword, control_keyword, target_display, control_display, cand_std, cand_alt, cand_hyper, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", item)
    conn.commit()
    conn.close()

init_db()

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
    t_start, t_end = None, None
    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            if keyword in w["word"].replace(" ", "").strip():
                t_start = float(w["start"])
                t_end = float(w["end"])
                break
        if t_start is not None:
            break
    sound = parselmouth.Sound(save_path)
    total_len = float(sound.get_total_duration())
    if t_start is None or t_end is None:
        return sound, 0.0, total_len
    scale = 0.68 if is_three else 1.0
    t_end = t_start + (t_end - t_start) * scale
    p_start = max(0.0, t_start - 0.03)
    p_end = min(total_len, t_end + 0.03)
    part = sound.extract_part(from_time=p_start, to_time=p_end, preserve_times=False)
    return part, t_start, t_end

def compute_phonetic_metrics(sound, gender="남성"):
    max_f = 5000.0 if gender == "남성" else 5500.0
    formants = sound.to_formant_burg(max_number_of_formants=5.0, maximum_formant=max_f)
    spectrogram = sound.to_spectrogram(window_length=0.005)
    times = formants.ts()
    
    f3_vals = []
    for t in times:
        val = formants.get_value_at_time(3, t)
        f3_vals.append(val)
        
    dur = float(sound.get_total_duration())
    t1 = dur * 0.4
    t2 = dur * 0.85
    
    offset_f3 = []
    for t in times:
        if t1 <= t <= t2:
            fv = formants.get_value_at_time(3, t)
            if not np.isnan(fv):
                offset_f3.append(fv)
                
    min_off = float(np.min(offset_f3)) if len(offset_f3) > 0 else 0.0
    return {"sound": sound, "spectrogram": spectrogram, "times": times, "f3": f3_vals, "min_off": min_off, "duration": dur}

def classify_pronunciation(d_t, d_c, s_info):
    drop = d_c["min_off"] - d_t["min_off"]
    denom = d_c["duration"] + 0.000001
    ratio = d_t["duration"] / denom
    if drop >= 70.0 and ratio >= 1.06:
        verdict = s_info["cand_hyper"]
        desc = "모음 말단 F3 하강과 지속시간 연장이 확인되어 '이중조음형'으로 판정되었습니다."
    elif drop >= 160.0 and ratio < 1.06:
        verdict = s_info["cand_alt"]
        desc = "폐쇄음이 약화되고 유음 성분이 뚜렷하여 '[ㄹ] 선택형 단순화'로 판정되었습니다."
    elif drop >= 70.0 and ratio < 1.06:
        verdict = s_info["cand_hyper"]
        desc = "모음 말단 F3 궤적에서 유음 잔여 효과가 포착되어 '이중조음형'으로 판정되었습니다."
    else:
        verdict = s_info["cand_std"]
        desc = "대조군과 F3 궤적이 일치하여 단일 폐쇄음으로 단순화되었습니다."
    return verdict, drop, ratio, desc

def plot_phonetics(snd_t, snd_c, pf_t, pf_c, w_t, w_c):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.5), sharey="row")
    sg_t, sg_c = pf_t["spectrogram"], pf_c["spectrogram"]
    axes[0, 0].pcolormesh(sg_t.x_grid(), sg_t.y_grid(), 10 * np.log10(sg_t.values), cmap="viridis", shading="auto")
    axes[0, 0].plot(pf_t["times"], pf_t["f3"], color="red", linewidth=2.5, label="F3 Track")
    axes[0, 0].set_title(f"A. 표적 어절: '{w_t}'")
    axes[0, 0].set_ylim(0, 4500)
    axes[0, 0].set_ylabel("Frequency (Hz)")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].pcolormesh(sg_c.x_grid(), sg_c.y_grid(), 10 * np.log10(sg_c.values), cmap="viridis", shading="auto")
    axes[0, 1].plot(pf_c["times"], pf_c["f3"], color="red", linewidth=2.5, label="F3 Track")
    axes[0, 1].set_title(f"B. 대조 어절: '{w_c}'")
    axes[0, 1].set_ylim(0, 4500)
    axes[0, 1].legend(loc="upper right")
    axes[1, 0].plot(snd_t.xs(), snd_t.values.T, color="#333")
    axes[1, 0].set_title(f"'{w_t}' 파형 ({pf_t['duration']:.2f}s)")
    axes[1, 1].plot(snd_c.xs(), snd_c.values.T, color="#005588")
    axes[1, 1].set_title(f"'{w_c}' 파형 ({pf_c['duration']:.2f}s)")
    plt.tight_layout()
    return fig

st.sidebar.title("메뉴 선택")
app_mode = st.sidebar.radio("모드 선택", ["학생 발음 실험 참여", "교수/연구자 관리자 모드"])

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
            if not st_name.strip():
                st.error("⚠️ 상단 1단계에서 이름을 입력해 주세요.")
            elif st.button("🚀 녹음 완료 및 음향 분석 실행", key=f"btn_eval_{step_idx}", use_container_width=True):
                with st.spinner("단어 정렬 및 음향 지표 산출 중..."):
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pt = os.path.join(AUDIO_DIR, f"{st_name}_{cur_set['target_display']}_{stamp}.wav")
                    pc = os.path.join(AUDIO_DIR, f"{st_name}_{cur_set['control_display']}_{stamp}.wav")
                    is_3 = ("밟" in cur_set['target_keyword'])

                    snd_t, st_t, ed_t = extract_aligned_audio(st.session_state[k_t], cur_set["target_keyword"], pt, is_3)
                    snd_c, st_c, ed_c = extract_aligned_audio(st.session_state[k_c], cur_set["control_keyword"], pc, is_3)
                    pf_t = compute_phonetic_metrics(snd_t, st_gender)
                    pf_c = compute_phonetic_metrics(snd_c, st_gender)
                    verdict, drop, ratio, desc = classify_pronunciation(pf_t, pf_c, cur_set)

                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""INSERT INTO participant_results (created_at, student_name, gender, age, hometown, set_name, target_display, control_display, f3_drop, duration_target, duration_control, closure_ratio, classified_label, audio_path_target, audio_path_control) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), st_name.strip(), st_gender, st_age, st_region, cur_set['set_name'], cur_set['target_display'], cur_set['control_display'], drop, pf_t["duration"], pf_c["duration"], ratio, verdict, pt, pc))
                    conn.commit()
                    conn.close()

                    st.session_state[f"eval_{step_idx}"] = {"verdict": verdict, "drop": drop, "ratio": ratio, "desc": desc, "snd_t": snd_t, "snd_c": snd_c, "pf_t": pf_t, "pf_c": pf_c, "set": cur_set}

        if f"eval_{step_idx}" in st.session_state:
            ev = st.session_state[f"eval_{step_idx}"]
            st.markdown("---")
            st.subheader("🎯 현실 발음 변이형 대조 분석")
            cands = [("표준 발음형", ev['set']['cand_std']), ("방언/비표준 변이형", ev['set']['cand_alt']), ("이중조음/과도교정", ev['set']['cand_hyper'])]
            cols = st.columns(3)
            for i, (lab, val) in enumerate(cands):
                with cols[i]:
                    if val == ev['verdict']:
                        st.markdown(f"""<div style="border: 3px solid #FF4B4B; background: rgba(255,75,75,0.08); padding: 15px; border-radius: 10px; text-align: center;"><b style="color: #FF4B4B;">👉 피험자 발음 일치</b><div style="font-size: 24px; font-weight: 900; margin: 4px 0;">{val}</div><small style="color: #555;">({lab})</small></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div style="border: 1px solid #DDD; background: #FAFAFA; padding: 15px; border-radius: 10px; text-align: center; opacity: 0.65;"><small style="color: #888;">후보군</small><div style="font-size: 17px; margin: 4px 0;">{val}</div><small style="color: #999;">({lab})</small></div>""", unsafe_allow_html=True)

            st.write(f"💡 **분석 해설:** {ev['desc']}")
            st.info(f"📘 **관련 음운 규칙:** {ev['set']['description']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("최종 판정형", ev['verdict'])
            m2.metric("모음 말단 F3 하강치", f"{int(ev['drop'])} Hz")
            m3.metric("지속시간 비율", f"{ev['ratio']:.2f} 배")
            st.pyplot(plot_phonetics(ev['snd_t'], ev['snd_c'], ev['pf_t'], ev['pf_c'], ev['set']['target_display'], ev['set']['control_display']))

            nxt = f"다음 실험({step_idx + 2}단계)으로 넘어가기 ➡️" if step_idx + 1 < total_sets else "모든 실험 완료하기 🏁"
            if st.button(nxt, type="primary", use_container_width=True):
                st.session_state.current_step += 1
                st.rerun()
    else:
        st.progress(1.0, text="모든 실험 세트 완료!")
        st.success("🎉 준비된 모든 실험 세트의 녹음과 분석이 성공적으로 끝났습니다. 참여 감사합니다!")
        if st.button("🔄 처음부터 다시 하기"):
            st.session_state.current_step = 0
            st.rerun()

elif app_mode == "교수/연구자 관리자 모드":
    st.title("🔒 국어음운론 연구 관리자 시스템")
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        st.subheader("🔑 관리자 로그인")
        aid = st.text_input("관리자 아이디")
        apw = st.text_input("비밀번호", type="password")
        if st.button("로그인", type="primary", use_container_width=True):
            if aid == "professor" and apw == "linguist2026":
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
    else:
        c_top1, c_top2 = st.columns([8, 2])
        c_top1.write("👋 교수님 관리자 계정으로 접속 중입니다.")
        if c_top2.button("로그아웃", use_container_width=True):
            st.session_state.admin_logged_in = False
            st.rerun()

        t1, t2 = st.tabs(["📊 전체 발음 통계 분석", "🎧 참여자 데이터 상세 분석 및 삭제"])
        conn = sqlite3.connect(DB_PATH)
        df_all = pd.read_sql_query("SELECT * FROM participant_results ORDER BY id DESC", conn)
        conn.close()

        with t1:
            if df_all.empty:
                st.info("수집된 학생 데이터가 없습니다.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("총 분석 발화 건수", f"{len(df_all)} 건")
                m2.metric("참여 학생 수", f"{df_all['student_name'].nunique()} 명")
                std_r = (df_all['classified_label'].str.contains("표준")).mean() * 100
                m3.metric("표준 규칙 실현율", f"{std_r:.1f} %")
                st.bar_chart(df_all['classified_label'].value_counts())
                st.dataframe(pd.crosstab(df_all['hometown'], df_all['classified_label'], margins=True), use_container_width=True)

        with t2:
            if df_all.empty:
                st.info("수집된 데이터가 없습니다.")
            else:
                st.download_button("💾 전체 CSV 다운로드", df_all.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'), f"corpus_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
                st.dataframe(df_all[['id', 'created_at', 'student_name', 'gender', 'age', 'hometown', 'set_name', 'classified_label', 'f3_drop', 'closure_ratio']], use_container_width=True)

                st.markdown("---")
                st.subheader("🔍 개별 참여자 상세 음향 분석 및 음성 청취")
                pid = st.selectbox("조회 대상 선택:", df_all['id'].tolist(), format_func=lambda x: f"ID {x} | {df_all.loc[df_all['id']==x, 'student_name'].values[0]} ({df_all.loc[df_all['id']==x, 'hometown'].values[0]}) - {df_all.loc[df_all['id']==x, 'set_name'].values[0]} ➔ {df_all.loc[df_all['id']==x, 'classified_label'].values[0]}")
                r_sel = df_all[df_all['id'] == pid].iloc[0]

                lbl = str(r_sel['classified_label'])
                if "이중조음" in lbl:
                    theme_col, bg_col = "#E05638", "rgba(224, 86, 56, 0.08)"
                elif "표준" in lbl:
                    theme_col, bg_col = "#2E7D32", "rgba(46, 125, 50, 0.08)"
                else:
                    theme_col, bg_col = "#1565C0", "rgba(21, 101, 192, 0.08)"

                st.markdown(f"""
                <div style="border-left: 6px solid {theme_col}; background: {bg_col}; padding: 16px 20px; border-radius: 8px; margin: 12px 0 16px 0;">
                    <div style="font-size: 14px; color: #555; font-weight: 600;">실험 과제: {r_sel['set_name']}</div>
                    <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 6px;">
                        <span style="font-size: 26px; font-weight: 900; color: {theme_col};">{lbl}</span>
                        <span style="font-size: 15px; color: #333;">(참가자: <b>{r_sel['student_name']}</b> | {r_sel['gender']}, {r_sel['age']}세, {r_sel['hometown']})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                f_drop = float(r_sel['f3_drop'])
                c_ratio = float(r_sel['closure_ratio'])
                
                sim_std = max(5, int(100 - min(90, abs(f_drop) * 0.45 + max(0, c_ratio - 1.0) * 120)))
                sim_hyper = max(5, int(min(95, max(0, f_drop) * 0.55 + max(0, c_ratio - 1.0) * 80)))
                sim_alt = max(5, int(min(95, max(0, f_drop) * 0.65 - max(0, c_ratio - 1.1) * 40)))

                st.markdown("##### 🎯 발음형별 음향적 근접도(유사도)")
                sc1, sc2, sc3 = st.columns(3)
                sc1.progress(sim_std / 100, text=f"표준형 일치도: {sim_std}%")
                sc2.progress(sim_hyper / 100, text=f"이중조음형 근접도: {sim_hyper}%")
                sc3.progress(sim_alt / 100, text=f"비표준(단순화) 근접도: {sim_alt}%")

                st.info(f"💡 **음향학적 근접 분석 해설:** F3 하강치가 **{int(f_drop)} Hz**, 조음 지속시간 비율이 **{c_ratio:.2f}배**로 측정되었습니다. " +
                        (f"모음 말단에서 유음 [ㄹ] 성분의 F3 하강과 함께 폐쇄 구간 지연이 동시에 뚜렷하게 관찰되어, 대조군 대비 **이중조음형({sim_hyper}%)에 가장 강하게 근접**합니다." if "이중조음" in lbl else
                         f"대조군 파형과 포먼트 궤적이 거의 일치하여, 겹받침이 잔여 성분 없이 **표준 단일 자음 단순화({sim_std}%)**에 가장 가깝게 실현되었습니다." if "표준" in lbl else
                         f"폐쇄음의 조음 제스처가 약화되고 유음의 공명 성분이 두드러져 **비표준 단순화형({sim_alt}%)**에 가깝습니다."))

                ca1, ca2 = st.columns(2)
                with ca1:
                    st.caption(f"🎧 표적 어절 ('{r_sel['target_display']}') 음성:")
                    if os.path.exists(r_sel['audio_path_target']): st.audio(r_sel['audio_path_target'])
                    else: st.caption("오디오 파일 없음")
                with ca2:
                    st.caption(f"🎧 대조 어절 ('{r_sel['control_display']}') 음성:")
                    if os.path.exists(r_sel['audio_path_control']): st.audio(r_sel['audio_path_control'])
                    else: st.caption("오디오 파일 없음")

                if os.path.exists(r_sel['audio_path_target']) and os.path.exists(r_sel['audio_path_control']):
                    with st.expander("📈 개별 스펙트로그램 및 파형 비교 그래프 열람", expanded=True):
                        try:
                            s_t = parselmouth.Sound(r_sel['audio_path_target'])
                            s_c = parselmouth.Sound(r_sel['audio_path_control'])
                            p_t = compute_phonetic_metrics(s_t, r_sel['gender'])
                            p_c = compute_phonetic_metrics(s_c, r_sel['gender'])
                            fig_adm = plot_phonetics(s_t, s_c, p_t, p_c, r_sel['target_display'], r_sel['control_display'])
                            st.pyplot(fig_adm)
                        except Exception as e:
                            st.caption(f"그래프 렌더링 오류: {e}")

                st.markdown("---")
                st.subheader("🗑️ 데이터 선택 삭제 (다중 선택)")
                opts = {r['id']: f"ID {r['id']} | {r['student_name']} ({r['hometown']}) - {r['classified_label']}" for _, r in df_all.iterrows()}
                sel_ids = st.multiselect("삭제 대상 선택:", list(opts.keys()), format_func=lambda x: opts[x])
                if sel_ids:
                    chk = st.checkbox(f"선택한 {len(sel_ids)}건의 데이터 및 음성 파일을 영구 삭제합니다.")
                    if st.button(f"🚨 선택된 {len(sel_ids)}건 삭제 실행", type="primary", disabled=not chk, use_container_width=True):
                        for did in sel_ids:
                            t_r = df_all[df_all['id'] == did]
                            if not t_r.empty:
                                r_v = t_r.iloc[0]
                                if os.path.exists(r_v['audio_path_target']): os.remove(r_v['audio_path_target'])
                                if os.path.exists(r_v['audio_path_control']): os.remove(r_v['audio_path_control'])
                        conn = sqlite3.connect(DB_PATH)
                        phs = ",".join(["?"] * len(sel_ids))
                        conn.cursor().execute(f"DELETE FROM participant_results WHERE id IN ({phs})", sel_ids)
                        conn.commit()
                        conn.close()
                        st.success("삭제 완료!")
                        st.rerun()
