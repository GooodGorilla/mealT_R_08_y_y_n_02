# ---------------------------------------------------------------------------
# Streamlit Web App: Meal Detection AI (Session State 적용 버전)
# ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import os
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Meal Detection AI", layout="wide")

st.title("🍽️ Meal Detection AI Analysis")
st.markdown("""
이 앱은 학습된 AI 모델을 사용하여 생체 데이터에서 **식사 구간(Meal Intervals)**을 자동으로 탐지합니다.
업로드된 파일(`csv`)을 분석하여 식사 확률, 예상 구간, 그리고 상세 리포트를 제공합니다.
""")

# --- [중요] 학습 파라미터 고정 (JSON 파일 대체) ---
FIXED_CONFIG = {
    'eval_window_size': 12,  # 60분
    'sub_window_size': 6,    # 30분
    'baseline_points': 4,    # 20분
    'stride': 1              # 5분
}

# --- 2. 파일 자동 탐색 함수 ---
def find_latest_file(pattern, description):
    files = glob.glob(pattern)
    if not files:
        st.error(f"❌ {description} 파일을 찾을 수 없습니다. (패턴: `{pattern}`)")
        return None
    return max(files, key=os.path.getmtime)

# --- 3. 리소스 로드 ---
@st.cache_resource
def load_resources():
    model_file = find_latest_file("trained_model_*.h5", "AI 모델")
    if model_file is None: return None, None, None
    
    try:
        model = tf.keras.models.load_model(model_file)
    except Exception as e:
        st.error(f"모델 로드 오류 ({model_file}): {e}")
        return None, None, None

    scaler_file = find_latest_file("scaler_*.pkl", "스케일러")
    if scaler_file is None: return None, None, None
        
    try:
        scaler = joblib.load(scaler_file)
    except Exception as e:
        st.error(f"스케일러 로드 오류 ({scaler_file}): {e}")
        return None, None, None

    filenames = {
        "model": os.path.basename(model_file),
        "scaler": os.path.basename(scaler_file)
    }

    return model, scaler, filenames

resources = load_resources()
if resources is None or resources[0] is None:
    st.warning("필요한 파일(.h5, .pkl)이 GitHub 저장소에 있는지 확인해주세요.")
    st.stop()

model, scaler_global, filenames = resources

# --- 4. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정")
st.sidebar.success(f"✅ 모델 로드: `{filenames['model']}`")
st.sidebar.info(f"""
**🔒 적용된 학습 파라미터**
- 평가 구간: {FIXED_CONFIG['eval_window_size']*5}분
- 서브 윈도우: {FIXED_CONFIG['sub_window_size']*5}분
- 기준 체온 구간: {FIXED_CONFIG['baseline_points']*5}분
- Stride: {FIXED_CONFIG['stride']*5}분
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ 민감도 조절")

temp_change_limit = st.sidebar.slider("급변 체온 기준 (°C)", 1.0, 2.0, 1.2, 0.1)
prob_threshold = st.sidebar.slider("서브윈도우 판정 기준 (Prob)", 0.1, 0.9, 0.5, 0.05)
# [수정] 윈도우 식사 비율 기본값 0.3 -> 0.2 변경
window_meal_threshold = st.sidebar.slider("윈도우 식사 비율 (Ratio)", 0.1, 0.9, 0.2, 0.05)
gt_threshold = st.sidebar.slider("정답 라벨 기준 (GT)", 0.1, 0.9, 0.5, 0.05)

# --- 5. 헬퍼 함수 ---
def minutes_to_time(minutes):
    hours = int(minutes // 60) % 24 
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

def handle_missing_times(df):
    if 'Time(min)' not in df.columns and '24h_time' in df.columns:
        try:
            time_parts = df['24h_time'].str.split(':', expand=True).astype(int)
            df['Time(min)'] = time_parts[0] * 60 + time_parts[1]
        except: return None, False
    elif 'Time(min)' not in df.columns: return None, False

    all_subjects_filled_dfs = []
    for subject_id, group in df.groupby('Subject_ID'):
        group = group.sort_values(by='Time(min)').drop_duplicates(subset=['Time(min)'])
        min_time, max_time = group['Time(min)'].min(), group['Time(min)'].max()
        master_df = pd.DataFrame(index=pd.RangeIndex(start=min_time, stop=max_time + 5, step=5, name='Time(min)'))
        group = group.set_index('Time(min)')
        merged = master_df.join(group, how='left')
        merged['is_interpolated'] = pd.isna(merged['Subject_ID']).astype(int)
        merged['Subject_ID'] = subject_id
        merged['Label_EN'] = merged['Label_EN'].fillna('Not-Meal (Filled)')
        if 'BodyTemp(°C)' in merged.columns: merged.rename(columns={'BodyTemp(°C)': 'BodyTemp'}, inplace=True)
        merged['BodyTemp'] = merged['BodyTemp'].interpolate(method='linear', limit_direction='both').ffill().bfill()
        for col in ['Age', 'Sex', 'Weight(kg)', 'Menstrual']:
            if col in merged.columns: merged[col] = merged[col].ffill().bfill()
        all_subjects_filled_dfs.append(merged.reset_index())
    return pd.concat(all_subjects_filled_dfs, ignore_index=True), True

def create_sequences_for_test(df, target, temp_change_limit, threshold_ratio_gt):
    X, y, times, subjects, is_interp, is_arti = [], [], [], [], [], []
    eval_win, stride, base_pts, sub_win = FIXED_CONFIG['eval_window_size'], FIXED_CONFIG['stride'], FIXED_CONFIG['baseline_points'], FIXED_CONFIG['sub_window_size']
    
    if 'BodyTemp(°C)' in df.columns: df.rename(columns={'BodyTemp(°C)': 'BodyTemp'}, inplace=True)
    df = df.sort_values(['Subject_ID', 'Time(min)'])
    
    unique_subs = df['Subject_ID'].unique()
    prog_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (sid, group) in enumerate(df.groupby('Subject_ID')):
        prog_bar.progress((idx + 1) / len(unique_subs))
        status_text.text(f"Processing Subject ID: {sid} ({idx+1}/{len(unique_subs)})")
        group = group.reset_index(drop=True)
        end_idx = len(group) - base_pts - eval_win + 1
        if end_idx <= 0: continue

        for i in range(0, end_idx, stride):
            base_data = group.iloc[i : i + base_pts]
            base_avg = base_data['BodyTemp'].mean()
            is_base_interp = (base_data['is_interpolated'] == 1).any()
            eval_data = group.iloc[i + base_pts : i + base_pts + eval_win]
            
            diffs = eval_data['BodyTemp'].values - base_avg
            is_artifact = np.any(np.abs(diffs) >= temp_change_limit)
            sub_diffs_list = []
            label_gt = 1 if (eval_data[target] == 1).mean() >= threshold_ratio_gt else 0
            is_eval_interp = (eval_data['is_interpolated'] == 1).any() or is_base_interp

            if not is_artifact:
                for j in range(0, eval_win - sub_win + 1, 1): 
                    sub_data = eval_data.iloc[j : j + sub_win]
                    feat = sub_data['BodyTemp'].values - base_avg
                    sub_diffs_list.append(feat.reshape(sub_win, 1))
            if is_artifact: X.append(np.zeros((0, sub_win, 1)))
            else: X.append(np.array(sub_diffs_list))
            y.append(label_gt); times.append(group['Time(min)'].iloc[i + base_pts + eval_win - 1])
            subjects.append(sid); is_interp.append(is_eval_interp); is_arti.append(is_artifact)
            
    prog_bar.empty()
    status_text.empty()
    return X, np.array(y), np.array(times), np.array(subjects), np.array(is_interp), np.array(is_arti), True

def get_interval_string(end_times, window_size_min):
    if len(end_times) == 0: return "없음"
    intervals = []
    offset = (window_size_min - 30) / 2
    for t in end_times:
        start = max(0, t - window_size_min)
        intervals.append([start + offset, start + offset + 30])
    intervals.sort(key=lambda x: x[0])
    merged = []
    if intervals:
        merged.append(intervals[0])
        for i in range(1, len(intervals)):
            if intervals[i][0] <= merged[-1][1] + 30: merged[-1][1] = max(merged[-1][1], intervals[i][1])
            else: merged.append(intervals[i])
    return ", ".join([f"{minutes_to_time(s)} ~ {minutes_to_time(e)}" for s, e in merged])

# --- 6. 분석 파이프라인 (Heavy Computation) ---
def run_analysis_pipeline(df_test, params):
    # 전처리
    with st.spinner("1/3 데이터 전처리 중..."):
        df_processed, success = handle_missing_times(df_test)
        if not success: return None

    target = 'Label_EN'
    if target not in df_processed.columns: df_processed[target] = 0
    df_processed[target] = df_processed[target].apply(lambda x: 1 if str(x).lower() == 'meal' else 0)

    # 시퀀스 생성
    with st.spinner("2/3 시퀀스 생성 중..."):
        X_list, y, ts, sids, is_int, is_art, ok = create_sequences_for_test(
            df_processed, target, params['temp_change_limit'], params['gt_threshold']
        )
    if not ok or len(X_list) == 0: return None

    # 예측 (가장 오래 걸림)
    with st.spinner("3/3 AI 모델 분석 중..."):
        preds, ratios = [], []
        # 배치 예측을 위해 데이터를 평탄화하면 좋겠지만, 로직 유지를 위해 루프 사용
        # (Streamlit에서는 진행률 보여주는게 더 나을 수 있음)
        prog_bar = st.progress(0)
        total = len(X_list)
        
        for i, x in enumerate(X_list):
            if i % 100 == 0: prog_bar.progress((i+1)/total)
            
            if is_art[i] or len(x) == 0:
                preds.append(2); ratios.append(0.0)
                continue
            
            N, T, F = x.shape
            x_scaled = scaler_global.transform(x.reshape(-1, F)).reshape(N, T, F)
            probs = model.predict(x_scaled, verbose=0).flatten()
            
            r = np.mean((probs >= params['prob_threshold']).astype(int))
            ratios.append(r)
            preds.append(1 if r >= params['window_meal_threshold'] else 0)
        
        prog_bar.empty()

    y_pred = np.array(preds)
    y_pred[is_int] = 2
    y_ratios = np.array(ratios)

    # 결과 패키징
    return {
        'df_processed': df_processed,
        'y_true': y,
        'y_pred': y_pred,
        'y_ratios': y_ratios,
        'times': ts,
        'subjects': sids,
        'params': params
    }

# --- 7. 결과 시각화 및 UI ---
uploaded_file = st.file_uploader("📂 테스트 데이터 파일 업로드 (CSV)", type=['csv'])

# 버튼 클릭 시: 분석 실행 및 결과 저장
if uploaded_file is not None:
    if st.button("🚀 분석 시작 (Run Analysis)"):
        params = {
            'temp_change_limit': temp_change_limit,
            'prob_threshold': prob_threshold,
            'window_meal_threshold': window_meal_threshold,
            'gt_threshold': gt_threshold
        }
        
        df_test = pd.read_csv(uploaded_file)
        results = run_analysis_pipeline(df_test, params)
        
        if results:
            st.session_state['analysis_results'] = results
            st.success("분석 완료! 아래에서 결과를 확인하세요.")
        else:
            st.error("분석 실패: 유효한 데이터가 없습니다.")

# 저장된 결과가 있으면 항상 표시 (버튼 안 눌러도 유지됨)
if 'analysis_results' in st.session_state:
    res = st.session_state['analysis_results']
    params = res['params'] # 분석 당시의 파라미터 사용
    
    st.divider()
    st.subheader("📊 분석 결과")
    
    # 전체 성능 지표
    valid_mask = (res['y_pred'] != 2)
    if np.sum(valid_mask) > 0:
        y_valid = res['y_true'][valid_mask]
        p_valid = res['y_pred'][valid_mask]
        
        acc = accuracy_score(y_valid, p_valid)
        f1 = f1_score(y_valid, p_valid, average='weighted', zero_division=0)
        try: auc = roc_auc_score(y_valid, res['y_ratios'][valid_mask])
        except: auc = 0.0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("정확도 (Accuracy)", f"{acc*100:.2f}%")
        col2.metric("F1 Score", f"{f1:.2f}")
        col3.metric("AUC", f"{auc:.2f}")
        
        with st.expander("상세 오차 행렬 보기"):
            cm = confusion_matrix(y_valid, p_valid, normalize='true', labels=[0, 1])
            st.dataframe(pd.DataFrame(cm*100, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1']).style.format("{:.1f}%"))
    
    # 개별 대상자 분석 (여기서 대상자를 바꿔도 분석 재실행 안 됨!)
    st.divider()
    st.subheader("📈 개별 대상자 상세 분석")
    
    u_ids = np.unique(res['subjects'])
    selected_subject = st.selectbox("대상자 선택:", u_ids)
    
    if selected_subject:
        mask = (res['subjects'] == selected_subject)
        sub_t = res['times'][mask]
        sub_r = res['y_ratios'][mask]
        sub_y = res['y_true'][mask]
        sub_p = res['y_pred'][mask]
        
        # 시간순 정렬
        sort_idx = np.argsort(sub_t)
        sub_t = sub_t[sort_idx]
        sub_r = sub_r[sort_idx]
        sub_y = sub_y[sort_idx]
        sub_p = sub_p[sort_idx]
        
        t_strs = [minutes_to_time(m) for m in sub_t]
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_strs, sub_r, label='Meal Ratio', color='royalblue')
        ax.axhline(y=params['window_meal_threshold'], color='red', linestyle='--', label='Threshold')
        
        ax.fill_between(t_strs, 0, 1, where=(sub_y==1), color='green', alpha=0.2, label='Actual Meal', transform=ax.get_xaxis_transform())
        ax.fill_between(t_strs, 0, 1, where=(sub_p==2), color='gray', alpha=0.5, label='Unknown', transform=ax.get_xaxis_transform(), step='post')
        
        ax.set_ylim(-0.05, 1.05)
        tick_idx = np.linspace(0, len(t_strs)-1, min(15, len(t_strs)), dtype=int)
        ax.set_xticks([t_strs[i] for i in tick_idx])
        plt.xticks(rotation=45)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.5)
        st.pyplot(fig)
        
        # 예측 구간 텍스트
        detected_times = sub_t[sub_p == 1]
        eval_win_min = FIXED_CONFIG['eval_window_size'] * 5
        st.write(f"🕒 **예측 식사 구간:** {get_interval_string(detected_times, eval_win_min)}")
        
        # 원본 데이터 확인
        with st.expander("원본 체온 데이터 보기"):
            raw_df = res['df_processed']
            sub_raw = raw_df[raw_df['Subject_ID'] == selected_subject].sort_values('Time(min)')
            sub_raw['Time_str'] = sub_raw['Time(min)'].apply(minutes_to_time)
            
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            ax2.plot(sub_raw['Time_str'], sub_raw['BodyTemp'], color='orange', label='BodyTemp')
            ax2.set_xticks(sub_raw['Time_str'][::max(1, len(sub_raw)//15)])
            plt.xticks(rotation=45)
            ax2.grid(True, alpha=0.5)
            st.pyplot(fig2)