import streamlit as st
from streamlit_mic_recorder import mic_recorder
import parselmouth, whisper, numpy as np, matplotlib.pyplot as plt
import pandas as pd, os, subprocess, socket, json
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 구글 API 통신 타임아웃 10초 설정
socket.setdefaulttimeout(10.0)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="국어음운론 현실 발음 조사 실습실", layout="wide")

AUDIO_DIR = "recordings_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Google Drive & Sheets 인증 연결 (PEM 줄바꿈 완벽 정규화)
@st.cache_resource
def get_gcp_clients():
    if "gcp" in st.secrets and "service_account_json" in st.secrets["gcp"]:
        creds_dict = json.loads(st.secrets["gcp"]["service_account_json"])
    else:
        creds_dict = dict(st.secrets["gcp_service_account"])

    # PEM 형식 정규화 처리 (줄바꿈 오류 완전 방지)
    if "private_key" in creds_dict:
        pk = creds_dict["private_key"]
        pk = pk.replace("\\n", "\n")
        lines = [line.strip() for line in pk.split("\n") if line.strip()]
        creds_dict["private_key"] = "\n".join(lines) + "\n"

    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets"
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    drive_service = build('drive', 'v3', credentials=creds)
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(st.secrets["storage"]["SHEET_ID"]).sheet1
    return drive_service, sheet

def save_to_cloud(pt, fn_t, pc, fn_c, row_base):
    """구글 드라이브 업로드 및 구글 시트 행 추가"""
    try:
        drive_service, sheet = get_gcp_clients()
        folder_id = st.secrets["storage"]["DRIVE_FOLDER_ID"]

        # 1. 표적 음원 업로드
        url_t = ""
        if os.path.exists(pt):
            media_t = MediaFileUpload(pt, mimetype='audio/wav', resumable=False)
            f_t = drive_service.files().create(
                body={'name': fn_t, 'parents': [folder_id]},
                media_body=media_t,
                fields='webViewLink'
            ).execute()
            url_t = f_t.get('webViewLink', '')

        # 2. 대조 음원 업로드
        url_c = ""
        if os.path.exists(pc):
            media_c = MediaFileUpload(pc, mimetype='audio/wav', resumable=False)
            f_c = drive_service.files().create(
                body={'name': fn_c, 'parents': [folder_id]},
                media_body=media_c,
                fields='webViewLink'
            ).execute()
            url_c = f_c.get('webViewLink', '')

        # 3. 구글 시트 행 추가
        row_data = row_base + [url_t, url_c]
        sheet.append_row(row_data)
        return True, "저장 성공"
    except Exception as e:
        return False, str(e)

def get_sheet_data():
    try:
        _, sheet = get_gcp_clients()
        records = sheet.get_all_records()
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()

SETS = [
    {"id": 1, "title": "과제 1: '낡지' vs '낙지'", "s_t": "신발이 너무 낡지 않았어?", "s_c": "낙지가 너무 맛있지 않아?", "kw_t": "낡지", "kw_c": "낙지", "disp_t": "낡지", "disp_c": "낙지", "cand": ["[낙찌] (표준: ㄱ단순화)", "[날찌] (비표준: ㄹ단순화)", "[낡찌] (과도교정: 이중조음)"], "rule": "표준어 규정 제11항: 어간 말 'ㄺ'은 자음 앞에서 [ㄱ]으로 발음"},
    {"id": 2, "title": "과제 2: '밟지' vs '밥지리'", "s_t": "내 발 좀 밟지 마.", "s_c": "벙어리를 밥지리라고도 한다.", "kw_t": "밟지", "kw_c": "밥지리", "disp_t": "밟지", "disp_c": "밥지", "cand": ["[밥찌] (표준: ㅂ단순화)", "[발찌] (일반화 오류: ㄹ단순화)", "[밟찌] (과도교정: 이중조음)"], "rule": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"},
    {"id": 3, "title": "과제 3: '밟도록' vs '밥도둑'", "s_t": "이 부분을 밟도록 해", "s_c": "간장게장을 밥도둑이라고 해", "kw_t": "밟도록", "kw_c": "밥도둑", "disp_t": "밟도", "disp_c": "밥도", "cand": ["[밥또록] (표준: ㅂ단순화)", "[발또록] (일반화 오류: ㄹ단순화)", "[밟또록] (과도교정: 이중조음)"], "rule": "표준어 규정 제10항 단서: 어간 '밟-'은 자음 앞에서 예외적으로 [ㅂ]으로 발음"}
]

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

    off_f3 = [fmt.get_value_at_time(3, t) for t in ts if dur * 0.45 <= t <= dur * 0.85 and not np.isnan(fmt.get_value_at_time(3, t))]
    min_off = float(np.min(off_f3)) if len(off_f3) > 0 else 0.0

    mid_f3 = [fmt.get_value_at_time(3, t) for t in ts if dur * 0.40 <= t <= dur * 0.65 and not np.isnan(fmt.get_value_at_time(3, t))]
    avg_mid = float(np.mean(mid_f3)) if len(mid_f3) > 0 else min_off

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
    gap_diff = m_c["avg_gap"] - m_t["avg_gap"]

    if "밟" in task["kw_t"]:
        internal_drop = m_t["avg_mid"] - m_t["min_off"]
        is_hyper = (diff_f3 >= 70.0) or (gap_diff >= 120.0) or (internal_drop >= 80.0 and ratio >= 0.96) or (drop_pct >= 0.6 and ratio >= 0.99)
        is_alt = (drop_pct >= 7.0 and ratio < 0.95 and gap_diff >= 200.0)

        if is_alt:
            return task["cand"][1], drop_pct, ratio, "양순 폐쇄음 성분이 실종되고 유음 성분이 지배적 ('[ㄹ] 선택형 단순화')"
        elif is_hyper:
            return task["cand"][2], drop_pct, ratio, "모음 말단 F3 하강 및 F2-F3 수렴 구조 감지 ('이중조음/과도교정형')"
        return task["cand"][0], drop_pct, ratio, "대조군과 음향 궤적 및 포먼트 간격 일치 (단일 양순 폐쇄음 규범 단순화)"
    else:
        if (drop_pct >= 1.6 and ratio >= 0.98) or diff_f3 >= 60.0:
            return task["cand"][2], drop_pct, ratio, "모음 말단 F3 상대 하강 뚜렷 ('이중조음형')"
        elif drop_pct >= 5.0 and ratio < 0.98:
            return task["cand"][1], drop_pct, ratio, "연구개 폐쇄음 약화 및 유음 성분 우세 ('[ㄹ] 선택형 단순화')"
        return task["cand"][0], drop_pct, ratio, "대조군 대비 변동폭 정상 범위 (단일 연구개 폐쇄음 규범 단순화)"

def plot_spec(m_t, m_c):
    fig, axes = plt.subplots(2, 2, figsize=(12, 5.5), sharey="row")
    labels = ["Target", "Control"]
    for idx, m in enumerate([m_t, m_c]):
        axes[0, idx].pcolormesh(m["sg"].x_grid(), m["sg"].y_grid(), 10 * np.log10(m["sg"].values), cmap="viridis", shading="auto")
        axes[0, idx].plot(m["ts"], m["f3"], color="red", linewidth=2.5, label="F3")
        axes[0, idx].set_title(labels[idx] + " Spectrogram", fontweight='bold')
        axes[0, idx].set_ylim(0, 4500)
        axes[1, idx].plot(m["snd"].xs(), m["snd"].values.T, color="#005588" if idx else "#222")
        axes[1, idx].set_title(labels[idx] + " Waveform (" + str(round(m['dur'], 2)) + "s)")
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
            st.warning("🗣️ 문장 1: " + task['s_t'])
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
            st.info("🗣️ 문장 2: " + task['s_c'])
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
            u_perc = st.radio("방금 읽으신 '" + task['disp_t'] + "'를 어떻게 발음했다고 느끼셨나요?", opts, key="p_" + str(step), horizontal=True)

            if not name.strip():
                st.error("1단계에서 이름을 입력해 주세요.")
            elif st.button("🚀 녹음 완료 및 음향 분석 실행", key="run_" + str(step), type="primary", use_container_width=True):
                with st.spinner("스펙트로그램 분석 및 클라우드 영구 저장 중..."):
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn_t = name + "_" + task['disp_t'] + "_" + stamp + ".wav"
                    fn_c = name + "_" + task['disp_c'] + "_" + stamp + ".wav"
                    pt = os.path.join(AUDIO_DIR, fn_t)
                    pc = os.path.join(AUDIO_DIR, fn_c)

                    s_t = slice_word(st.session_state[kt], task["kw_t"], pt, (task["id"] == 3))
                    s_c = slice_word(st.session_state[kc], task["kw_c"], pc, (task["id"] == 3))
                    m_t, m_c = get_metrics(s_t, gender), get_metrics(s_c, gender)
                    verdict, drop_pct, ratio, exp = classify(m_t, m_c, task)

                    row_base = [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        name.strip(), gender, int(age), region,
                        int(task["id"]), task["disp_t"],
                        round(float(drop_pct), 2), round(float(ratio), 2),
                        verdict, u_perc, ""
                    ]
                    cloud_ok, cloud_msg = save_to_cloud(pt, fn_t, pc, fn_c, row_base)

                    st.session_state["res_" + str(step)] = {
                        "verdict": verdict, "drop_pct": drop_pct, "ratio": ratio,
                        "exp": exp, "perc": u_perc, "m_t": m_t, "m_c": m_c,
                        "task": task, "cloud_ok": cloud_ok, "cloud_msg": cloud_msg
                    }

        if ("res_" + str(step)) in st.session_state:
            r = st.session_state["res_" + str(step)]
            st.markdown("---")
            
            if r.get("cloud_ok"):
                st.success("☁️ Google Drive 및 스프레드시트에 안전하게 저장되었습니다.")
            else:
                st.error("❌ 구글 클라우드 저장 실패 원인: " + str(r.get("cloud_msg")))

            st.subheader("🎯 스펙트로그램 음향 분석 결과")
            cols = st.columns(3)
            labs = ["표준 발음", "비표준/방언", "이중조음/과도교정"]
            for i, cand in enumerate(r["task"]["cand"]):
                with cols[i]:
                    if cand == r["verdict"]:
                        st.info("👉 음향 분석 판정형\n### " + str(cand) + "\n(" + labs[i] + ")")
                    else:
                        st.text("후보\n" + str(cand) + "\n(" + labs[i] + ")")

            is_eq = (r["perc"] in r["verdict"])
            if is_eq:
                st.success("✅ 자각 발음과 스펙트로그램 음향 판정이 일치합니다.\n• 자각: " + str(r['perc']) + " | • 판정: " + str(r['verdict']))
            else:
                st.warning("⚡ 자각 발음과 스펙트로그램 음향 판정 간 차이가 확인되었습니다.\n• 자각: " + str(r['perc']) + " | • 판정: " + str(r['verdict']))

            st.write("💡 **분석 해설:** " + r['exp'])
            st.info("📘 **관련 규정:** " + r['task']['rule'])
            st.pyplot(plot_spec(r["m_t"], r["m_c"]))

            if st.button("다음 과제로 넘어가기 ➡️" if step + 1 < len(SETS) else "모든 실습 완료 🏁", type="primary", use_container_width=True):
                st.session_state.step += 1
                st.rerun()
    else:
        st.success("🎉 모든 실습 과제를 마쳤습니다. 참여해 주셔서 감사합니다!")
        if st.button("🔄 처음부터 다시 하기"):
            st.session_state.step = 0
            st.rerun()

elif mode == "교수/연구자 관리자 모드":
    st.title("🔒 국어음운론 연구 관리자 시스템 (Cloud 연동)")
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

        t1, t2 = st.tabs(["📊 통계 분석", "🎧 참가자별 클라우드 데이터 확인"])
        df = get_sheet_data()

        with t1:
            if df.empty:
                st.info("Google Sheets에 수집된 데이터가 없습니다.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("총 발화 수", str(len(df)) + " 건")
                m2.metric("참여자 수", str(df['name'].nunique()) + " 명")
                ed = (df['expert'] != "").sum() if 'expert' in df.columns else 0
                m3.metric("연구자 판정 완료", str(ed) + " / " + str(len(df)) + " 건")

                st.write("▼ **학생 자각 발음 분포**")
                st.bar_chart(df['perceived'].value_counts())

        with t2:
            if df.empty:
                st.info("Google Sheets에 수집된 데이터가 없습니다.")
            else:
                st.download_button("💾 전체 CSV 다운로드", df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'), "corpus.csv", "text/csv")
                names = df['name'].unique().tolist()
                c_name = st.selectbox("조회할 참가자 선택:", names)
                u_df = df[df['name'] == c_name].sort_values(by="task_id", ascending=True)

                for _, row in u_df.iterrows():
                    task_idx = int(row['task_id']) - 1
                    t_info = SETS[task_idx]
                    
                    st.markdown("---")
                    st.markdown("### 📌 " + t_info["title"])
                    st.write("• 발화 일시: " + str(row['created_at']))
                    st.write("• 기계 판정: **" + str(row['classified']) + "** | 학생 자각: **" + str(row['perceived']) + "**")

                    col_u1, col_u2 = st.columns(2)
                    with col_u1:
                        if row.get('url_t'):
                            st.markdown("🔗 [Google Drive에서 표적 발화 듣기](" + str(row['url_t']) + ")")
                    with col_u2:
                        if row.get('url_c'):
                            st.markdown("🔗 [Google Drive에서 대조 발화 듣기](" + str(row['url_c']) + ")")
