import streamlit as st
import joblib
import os
import pandas as pd
import segmentation  # 분석 모듈 임포트

# --- 기존 챗봇 기능 함수 ---

def load_models(model_dir='models'):
    le = joblib.load(os.path.join(model_dir, 'le.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    rf = joblib.load(os.path.join(model_dir, 'rf.joblib'))
    return le, scaler, rf

def predict_genre(le, scaler, rf, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = rf.predict(input_scaled)
    genre = le.inverse_transform(prediction)
    return genre[0], prediction[0]

def add_message_bundle(user_message, chatbot_message, img_path):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    bundle = [
        {"role": "user", "text": user_message},
        {"role": "ai", "text": chatbot_message},
        {"role": "image", "text": img_path},
    ]
    st.session_state["chat_history"].insert(0, bundle)

def clear_chat_history():
    st.session_state["chat_history"] = []

# --- 페이지 렌더링 함수 ---

def render_chatbot_page():
    """방송 장르 예측 챗봇 페이지를 렌더링합니다."""
    st.title("방송 장르 예측 챗봇")
    st.header("분류: 스포츠, 애니메이션, 키즈, 홈쇼핑")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("입력")
        st.markdown("**입력값을 슬라이더로 조정하세요.**")
        avrg_wtchng_co = st.slider("하루 평균 시청 프로그램 수", 0, 20, 2, step=1)
        dawn_avrg_wtchng_time_co = st.slider("새벽 평균 시청 시간 (분)", 0, 180, 10, step=1)
        am_avrg_wtchng_time_co = st.slider("오전 평균 시청 시간 (분)", 0, 180, 40, step=1)
        pm_avrg_wtchng_time_co = st.slider("오후 평균 시청 시간 (분)", 0, 180, 30, step=1)
        evening_avrg_wtchng_time_co = st.slider("저녁 평균 시청 시간 (분)", 0, 180, 20, step=1)

        input_data = [avrg_wtchng_co, dawn_avrg_wtchng_time_co, am_avrg_wtchng_time_co, pm_avrg_wtchng_time_co, evening_avrg_wtchng_time_co]

        col1_a, col1_b = st.columns(2)
        with col1_a:
            if st.button("채팅 추가"):
                user_message = f"저는 하루 평균 {avrg_wtchng_co}개의 프로그램을 시청하고, 시간대별 시청 시간은 새벽 {dawn_avrg_wtchng_time_co}분, 오전 {am_avrg_wtchng_time_co}분, 오후 {pm_avrg_wtchng_time_co}분, 저녁 {evening_avrg_wtchng_time_co}분입니다."
                le, scaler, rf = load_models()
                predicted_genre, label = predict_genre(le, scaler, rf, input_data)
                chatbot_message = f"예상되는 방송 장르는 '{predicted_genre}'입니다!"
                img_path = os.path.join("img", f"{label}.png")
                add_message_bundle(user_message, chatbot_message, img_path)
        with col1_b:
            if st.button("전체 채팅 삭제"):
                clear_chat_history()

    with col2:
        st.header("챗봇 대화")
        for bundle in st.session_state.get("chat_history", []):
            for chat in bundle:
                role, msg = chat["role"], chat["text"]
                if role == "image":
                    st.image(msg, width=200)
                else:
                    st.chat_message(name=role).write(msg)

def render_segmentation_page():
    """지역 기반 고객군 분석 페이지를 렌더링합니다."""
    st.title("지역 기반 고객군 분석 (Region-based Segmentation)")

    if 'segmentation_data' not in st.session_state:
        st.session_state['segmentation_data'] = None

    with st.expander("분석 설명", expanded=False):
        st.write("""
        이 분석은 LG U+ IPTV 시청 데이터를 기반으로, 지역(행정동)들의 시청 패턴 유사성을 분석하여 고객군을 나눕니다.
        - **분석 방법**: K-Means Clustering
        - **주요 피처**: 하루 평균 시청 프로그램 수, 새벽/오전/오후/저녁 평균 시청 시간
        """)

    if st.button("분석 실행"):
        with st.spinner('데이터 로딩 및 최적 클러스터 분석 중...'):
            df = segmentation.load_data_from_db()
            if not df.empty:
                _, scaled_features, _, _ = segmentation.preprocess_for_segmentation(df)
                elbow_path = segmentation.find_optimal_clusters(scaled_features)
                st.session_state['segmentation_data'] = {
                    'df': df,
                    'elbow_chart_path': elbow_path
                }
            else:
                st.error("데이터를 불러오는 데 실패했습니다. DB 연결을 확인하세요.")
    
    if st.session_state['segmentation_data']:
        data = st.session_state['segmentation_data']
        st.subheader("1. 최적 클러스터 개수(K) 찾기 - Elbow Method")
        st.image(data['elbow_chart_path'])
        st.info("차트에서 팔꿈치(Elbow)처럼 급격히 꺾이는 지점이 최적의 K값입니다. 보통 3~5 사이에서 선택합니다.")

        st.subheader("2. K-Means 클러스터링 실행")
        k = st.number_input("클러스터 개수(K)를 입력하세요.", min_value=2, max_value=10, value=4, step=1)
        
        if st.button("클러스터링 결과 보기"):
            with st.spinner(f'{k}개의 클러스터로 분석 중...'):
                region_df, personas, radar_path = segmentation.run_segmentation(data['df'], n_clusters=k)
                
                st.subheader("3. 클러스터별 페르소나 분석")
                for i, p in personas.items():
                    st.markdown(f"- **Cluster {i}**: {p}")
                
                st.subheader("4. 클러스터별 시청 패턴 시각화")
                st.image(radar_path)

                st.subheader("5. 지역별 클러스터 할당 결과")
                st.dataframe(region_df[['CTPRVN_NM', 'SIGNGU_NM', 'ADSTRD_NM', 'cluster']])

# --- 메인 앱 실행 ---
def main():
    st.sidebar.title("메뉴")
    page = st.sidebar.radio("페이지 선택", ["방송 장르 예측 챗봇", "지역 기반 고객군 분석"])

    if page == "방송 장르 예측 챗봇":
        render_chatbot_page()
    elif page == "지역 기반 고객군 분석":
        render_segmentation_page()

if __name__ == "__main__":
    main()
