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
    {"id": 2, "title": "과제 2: '밟지' vs '밥지리'", "s_t": "내 발 좀 밟지 마.", "s_c": "벙어리를 밥지리라고도 한다.", "kw_t": "밟지", "kw_c": "밥지리", "disp_t": "밟지", "disp_c": "밥지", "cand": ["[밥찌] (표준: ㅂ단순화)", "[발찌] (일반화 오류: ㄹ단순화)", "[밟찌] (과도교정: 이중조음)"], "rule": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"},
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
    with open(tmp, "wb") as f:
        f.write(audio_bytes)
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(tmp):
        os.remove(tmp)

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
        if t0 is not None:
            break

    if (t0 is None) or not (0.18 <= (t1 - t0) <= 0.95):
        pos = 0.35 if "낡" in kw else (0.08 if "낙" in kw else (0.38 if "밟지" in kw else (0.42 if "밥지" in kw else 0.30)))
        t0, t1 = dur * pos, dur * pos + 0.42

    if is_3:
        t1 = t0 + (t1 - t0) * 0.68
    return snd.extract_part(from_time=max(0.0, t0 - 0.02), to_time=min(dur, t1 + 0.02), preserve_times=False)

def get_metrics(snd, gender="남성"):
    max_f = 5000.0 if gender == "남성" else 5500.0
    fmt = snd.to_formant_burg(max_number_of_formants=5.0, maximum_formant=max_f)
    sg = snd.to_spectrogram(window_length=0.005)
    dur = float(snd.get_total_duration())
    ts = fmt.ts()
    f3s = [fmt.get_value_at_time(3, t) for t in ts]
    f2s = [fmt.get_value_at_time(2, t) for t in ts]

    # 모음 후반부 F3 최솟값
    off_f3 = [fmt.get_value_at_time(3, t) for t in ts if dur * 0.45 <= t <= dur * 0.85 and not np.isnan(fmt.get_value_at_time(3, t))]
    min_off = float(np.min(off_f3)) if len(off_f3) > 0 else 0.0

    # 모음 중앙부 F3 평균
    mid_f3 = [fmt.get_value_at_time(3, t) for t in ts if dur * 0.40 <= t <= dur * 0.65 and not np.isnan(fmt.get_value_at_time(3, t))]
    avg_mid = float(np.mean(mid_f3)) if len(mid_f3) > 0 else min_off

    # 모음 후반부 F3 - F2 간격(Band gap) 평균 계산 (유음성 수렴도 지표)
    gaps = []
    for t in ts:
        if dur * 0.45 <= t <= dur * 0.85:
            v2 = fmt.get_value_at_time(2, t)
            v3 = fmt.get_value_at_time(3, t)
            if not np.isnan(v2) and not np.isnan(v3):
                gaps.append(v3 - v2)
    avg_gap = float(np.mean(gaps)) if len(gaps) > 0 else 1200.0

    return {"snd": snd, "sg": sg, "ts": ts, "f3": f3s, "min_off": min_off, "avg_mid": avg_mid, "avg_gap": avg_gap, "dur": dur}

def classify(m_t, m_c, task):
    diff_f3 = m_c["min_off"] - m_t["min_off"]
    drop_pct = (diff_f3 / max(100.0, m_c["min_off"])) * 100.0
    ratio = m_t["dur"] / (m_c["dur"] + 1e-6)
    gap_diff = m_c["avg_gap"] - m_t["avg_gap"]  # 표적의 F3-F2 간격이 얼마나 더 좁혀졌는가

    if "밟" in task["kw_t"]:
        # 모음 자체 내에서의 하강폭
        internal_drop = m_t["avg_mid"] - m_t["min_off"]

        # 이중조음 판정 기준:
        # 1. 대조군 대비 F3가 70Hz 이상 낮거나,
        # 2. 대조군 대비 F3-F2 간격이 120Hz 이상 좁아지거나,
        # 3. 자체 내부 F3 하강이 80Hz 이상이면서 지속시간이 유사/연장될 때
        is_hyper = (diff_f3 >= 70.0) or (gap_diff >= 120.0) or (internal_drop >= 80.0 and ratio >= 0.96) or (drop_pct >= 0.6 and ratio >= 0.99)
        
        # 완전 유음화 판정 기준 (폐쇄음이 상실되고 ㄹ 성분만 남은 경우)
        is_alt = (drop_pct >= 7.0 and ratio < 0.95 and gap_diff >= 200.0)

        if is_alt:
            return task["cand"][1], drop_pct, ratio, "양순 폐쇄음 성분이 실종되고 유음 성분이 지배적 ('[ㄹ] 선택형 단순화')"
        elif is_hyper:
            return task["cand"][2], drop_pct, ratio, "모음 말단 F3 하강 및 F2-F3 수렴 구조 감지 ('이중조음/과도교정형')"
        return task["cand"][0], drop_pct, ratio, "대조군과 음향 궤적 및 포먼트 간격 일치 (단일 양순 폐쇄음 규범 단순화)"
    else:
        # '낡지' 계열 판정
        if (drop_pct >= 1.6 and ratio >= 0.98) or diff_f3 >= 60.0:
            return task["cand"][2], drop_pct, ratio, "모음 말단 F3 상대 하강 뚜렷 ('이중조음형')"
        elif drop_pct >= 5.0 and ratio < 0.98:
            return task["cand"][1], drop_pct, ratio, "연구개 폐쇄음 약화 및 유음 성분 우세 ('[ㄹ] 선택형 단순화')"
        return task["cand"][0], drop_pct, ratio, "대조군 대비 변동폭 정상 범위 (단일 연구개 폐쇄음 규범 단순화)"

def plot_spec(m_t, m_c, w_t, w_c):
    fig, axes = plt.subplots(2, 2, figsize=(12, 5.5), sharey="row")
    for idx, (m, w) in enumerate([(m_t, w_t), (m_c, w_c)]):
        axes[0, idx].pcolormesh(m["sg"].x_grid(), m["sg"].y_grid(), 10 * np.log10(m["sg"].values), cmap="viridis", shading="auto")
        axes[0, idx].plot(m["ts"], m["f3"], color="red", linewidth=2.5, label="F3")
        axes[0, idx].set_title(str(w) + " Spectrogram", fontweight='bold')
        axes[0, idx].set_ylim(0, 4500)
        axes[1, idx].plot(m["snd"].xs(), m["snd"].values.T, color="#005588" if idx else "#222")
        axes[1, idx].set_title(str(w) + " Waveform (" + str(round(m['dur'], 2)) + "s)")
    plt.tight_layout()
    return fig

st.sidebar.title("메뉴 선택")
mode = st.sidebar.radio("모드 선택", ["학생 발음 실험 참여", "교수/연구자 관리자 모드"])

if mode == "학생 발음 실험 참여":
    st.title("🎙️ 국어음운론 현실 발음 대조 실습")
    st.info("📢 평소 말씀하시는 습관대로 편안하게 문장을 읽고 녹음해 주세요.")

    with st.expander("👤 1단계: 참가자 기본 정보", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("이름 (또는 닉네임)")
        gender = c2.selectbox("성별", ["여성", "남성", "기타"])
        age = c3.number_input("나이(만)", 5, 90, 22)
        region = c4.selectbox("성장 지역", ["수도권", "경남(창원/마산/진주 등)", "부산", "대구", "경북", "충청", "전라", "강원", "제주", "기타"])

    if "step" not in st.session_state:
        st.session_state.step = 0
    step = st.session_state.step

    if step < len(SETS):
        task = SETS[step]
        st.progress(step / len(SETS), text="과제 " + str(step + 1) + " / " + str(len(SETS)))
        st.subheader(task["title"])

        kt, kc = "aud_t_" + str(task['id']), "aud_c_" + str(task['id'])
        if kt not in st.session_state:
            st.session_state[kt] = None
        if kc not in st.session_state:
            st.session_state[kc] = None

        col_t, col_c = st.columns(2)
        with col_t:
            st.warning("🗣️ **문장 1: " + task['s_t'] + "**")
            if st.session_state[kt] is None:
                rt = mic_recorder(start_prompt="🔴 문장 1 녹음", stop_prompt="⏹️ 완료", key="rt_" + str(task['id']) + "_" + str(step))
                if rt and rt.get("bytes"):
                    st.session_state[kt] = rt["bytes"]
                    st.rerun()
            else:
                st.audio(st.session_state[kt], format="audio/wav")
                if st.button("🔄 문장 1 다시 녹음", key="re_t_" + str(task['id'])):
                    st.session_state[kt] = None
                    st.rerun()

        with col_c:
            st.info("🗣️ **문장 2: " + task['s_c'] + "**")
            if st.session_state[kc] is None:
                rc = mic_recorder(start_prompt="🔴 문장 2 녹음", stop_prompt="⏹️ 완료", key="rc_" + str(task['id']) + "_" + str(step))
                if rc and rc.get("bytes"):
                    st.session_state[kc] = rc["bytes"]
                    st.rerun()
            else:
                st.audio(st.session_state[kc], format="audio/wav")
                if st.button("🔄 문장 2 다시 녹음", key="re_c_" + str(task['id'])):
                    st.session_state[kc] = None
                    st.rerun()

        if st.session_state[kt] and st.session_state[kc]:
            st.markdown("---")
            opts = [c.split(" ")[0] for c in task["cand"]]
            u_perc = st.radio("🤔 방금 읽으신 '" + task['disp_t'] + "'를 어떻게 발음했다고 느끼셨나요?", opts, key="p_" + str(step), horizontal=True)

            if not name.strip():
                st.error("⚠️ 1단계에서 이름을 입력해 주세요.")
            elif st.button("🚀 녹음 완료 및 음향 분석 실행", key="run_" + str(step), type="primary", use_container_width=True):
                with st.spinner("스펙트로그램 분석 중..."):
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pt = os.path.join(AUDIO_DIR, name + "_" + task['disp_t'] + "_" + stamp + ".wav")
                    pc = os.path.join(AUDIO_DIR, name + "_" + task['disp_c'] + "_" + stamp + ".wav")

                    s_t = slice_word(st.session_state[kt], task["kw_t"], pt, (task["id"] == 3))
                    s_c = slice_word(st.session_state[kc], task["kw_c"], pc, (task["id"] == 3))
                    m_t, m_c = get_metrics(s_t, gender), get_metrics(s_c, gender)
                    verdict, drop_pct, ratio, exp = classify(m_t, m_c, task)

                    conn = sqlite3.connect(DB_PATH)
                    conn.cursor().execute("""INSERT INTO results (created_at, name, gender, age, region, task_id, target_word, drop_pct, ratio, classified, perceived, expert, path_t, path_c)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name.strip(), gender, age, region, task["id"], task["disp_t"], drop_pct, ratio, verdict, u_perc, None, pt, pc))
                    conn.commit()
                    conn.close()

                    st.session_state["res_" + str(step)] = {"verdict": verdict, "drop_pct": drop_pct, "ratio": ratio, "exp": exp, "perc": u_perc, "m_t": m_t, "m_c": m_c, "task": task}

        if ("res_" + str(step)) in st.session_state:
            r = st.session_state["res_" + str(step)]
            st.markdown("---")
            st.subheader("🎯 스펙트로그램 음향 분석 결과")
            cols = st.columns(3)
            labs = ["표준 발음", "비표준/방언", "이중조음/과도교정"]
            for i, cand in enumerate(r["task"]["cand"]):
                with cols[i]:
                    if cand == r["verdict"]:
                        st.markdown("<div style='border: 3px solid #FF4B4B; background: rgba(255,75,75,0.08); padding: 12px; border-radius: 8px; text-align: center;'><b style='color:#FF4B4B;'>👉 음향 분석 판정형</b><div style='font-size:22px; font-weight:900;'>" + str(cand) + "</div><small>(" + labs[i] + ")</small></div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='border: 1px solid #DDD; background: #FAFAFA; padding: 12px; border-radius: 8px; text-align: center; opacity: 0.65;'><small>후보</small><div style='font-size:16px;'>" + str(cand) + "</div><small>(" + labs[i] + ")</small></div>", unsafe_allow_html=True)

            is_eq = (r["perc"] in r["verdict"])
            b_col = "#2E7D32" if is_eq else "#D97706"
            b_txt = "✅ 자각 발음과 스펙트로그램 음향 판정이 일치합니다." if is_eq else "⚡ 자각 발음과 스펙트로그램 음향 판정 간 차이가 확인되었습니다."
            st.markdown("<div style='border-left: 5px solid " + b_col + "; background: #F9FAFB; padding: 12px; border-radius: 6px; margin: 12px 0;'><b>" + b_txt + "</b><br><small>• 자각: <b>" + str(r['perc']) + "</b> | • 음향 판정: <b>" + str(r['verdict']) + "</b></small></div>", unsafe_allow_html=True)
            st.write("💡 **분석 해설:** " + r['exp'])
            st.info("📘 **관련 규정:** " + r['task']['rule'])
            st.pyplot(plot_spec(r["m_t"], r["m_c"], r["task"]["disp_t"], r["task"]["disp_c"]))

            if st.button("다음 과제로 넘어가기 ➡️" if step + 1 < len(SETS) else "모든 실습 완료 🏁", type="primary", use_container_width=True):
                st.session_state.step += 1
                st.rerun()
    else:
        st.success("🎉 모든 실습 과제를 마쳤습니다. 참여해 주셔서 감사합니다!")
        if st.button("🔄 처음부터 다시 하기"):
            st.session_state.step = 0
            st.rerun()

elif mode == "교수/연구자 관리자 모드":
    st.title("🔒 국어음운론 연구 관리자 시스템")
    if "admin" not in st.session_state:
        st.session_state.admin = False
    if not st.session_state.admin:
        aid, apw = st.text_input("아이디"), st.text_input("비밀번호", type="password")
        if st.button("로그인", type="primary"):
            if aid == "professor" and apw == "linguist2026":
                st.session_state.admin = True
                st.rerun()
            else:
                st.error("로그인 정보 불일치")
    else:
        if st.button("로그아웃"):
            st.session_state.admin = False
            st.rerun()

        t1, t2 = st.tabs(["📊 통계 분석", "🎧 참가자별 청취 판독"])
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)
        conn.close()

        with t1:
            if df.empty:
                st.info("수집된 데이터가 없습니다.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("총 발화 수", str(len(df)) + " 건")
                m2.metric("참여자 수", str(df['name'].nunique()) + " 명")
                ed = df['expert'].notnull().sum()
                m3.metric("연구자 판정 완료", str(ed) + " / " + str(len(df)) + " 건")

                st.write("▼ **학생 자각 발음 분포**")
                st.bar_chart(df['perceived'].value_counts())
                if ed > 0:
                    st.write("▼ **학생 자각(행) × 연구자 판단(열) 교차표**")
                    st.dataframe(pd.crosstab(df['perceived'], df['expert'], margins=True), use_container_width=True)

        with t2:
            if df.empty:
                st.info("수집된 데이터가 없습니다.")
            else:
                st.download_button("💾 전체 CSV 다운로드", df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'), "corpus.csv", "text/csv")
                names = df['name'].unique().tolist()
                c_name = st.selectbox("조회할 참가자 선택:", names)
                u_df = df[df['name'] == c_name].sort_values(by="task_id", ascending=True)

                cols = st.columns(len(u_df))
                for i, (_, row) in enumerate(u_df.iterrows()):
                    with cols[i]:
                        st.markdown("#### 📌 과제 " + str(row['task_id']))
                        st.write("• 기계 판정: **" + str(row['classified']) + "**\n• 학생 자각: **" + str(row['perceived']) + "**")
                        if row['expert'] and row['expert'] != "None":
                            st.success("👑 연구자 확정: " + str(row['expert']))

                        if os.path.exists(row['path_t']):
                            st.caption("표적 발화 음성:")
                            st.audio(row['path_t'])

                        cand_opts = SETS[int(row['task_id']) - 1]["cand"]
                        def_i = 0
                        for idx_c, c_text in enumerate(cand_opts):
                            if row['expert'] and row['expert'].split(" ")[0] in c_text:
                                def_i = idx_c
                                break

                        chosen_e = st.selectbox("판독 (ID " + str(row['id']) + ")", cand_opts, index=def_i, key="sel_" + str(row['id']))
                        if st.button("💾 연구자 판독 저장", key="btn_" + str(row['id'])):
                            conn = sqlite3.connect(DB_PATH)
                            conn.cursor().execute("UPDATE results SET expert = ? WHERE id = ?", (chosen_e, row['id']))
                            conn.commit()
                            conn.close()
                            st.rerun()

                st.markdown("---")
                del_btn_key = "btn_del_" + str(c_name)
                if st.button("🚨 '" + str(c_name) + "' 참가자 데이터 영구 삭제", key=del_btn_key, type="primary"):
                    for _, d_row in u_df.iterrows():
                        if os.path.exists(str(d_row['path_t'])):
                            try:
                                os.remove(str(d_row['path_t']))
                            except:
                                pass
                        if os.path.exists(str(d_row['path_c'])):
                            try:
                                os.remove(str(d_row['path_c']))
                            except:
                                pass

                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("DELETE FROM results WHERE name = ?", (str(c_name),))
                    conn.commit()
                    conn.close()
                    st.rerun()
