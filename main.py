# 모델을 joblib로 pkl dump, load 가능
import streamlit as st
import joblib
import os

# 모델 로드 함수
def load_models(model_dir='models'):
    le = joblib.load(os.path.join(model_dir, 'le.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    rf = joblib.load(os.path.join(model_dir, 'rf.joblib'))
    return le, scaler, rf

# 예측 함수
def predict_genre(le, scaler, rf, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = rf.predict(input_scaled)
    genre = le.inverse_transform(prediction)
    return genre[0], prediction[0]

# 채팅 추가 함수 (묶음 처리)
def add_message_bundle(user_message, chatbot_message, img_path):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    # 메시지 묶음을 리스트로 구성 후 맨 앞에 삽입
    bundle = [
        {"role": "user", "text": user_message},
        {"role": "ai", "text": chatbot_message},
        {"role": "image", "text": img_path},
    ]
    st.session_state["chat_history"].insert(0, bundle)

# 채팅 삭제 함수
def clear_chat_history():
    st.session_state["chat_history"] = []

# 메인 앱
def main():
    st.title("방송 장르 예측 챗봇")
    st.header("분류: 스포츠,애니메이션, 키즈, 홈쇼핑")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # 화면 분할
    col1, col2 = st.columns([2, 3])

    # 왼쪽: 값 입력
    with col1:
        st.header("입력")
        st.markdown("**입력값을 슬라이더로 조정하세요.**")

        avrg_wtchng_co = st.slider("하루 평균 시청 프로그램 수", 0, 20, 2, step=1)
        dawn_avrg_wtchng_time_co = st.slider("새벽 평균 시청 시간 (분)", 0, 180, 10, step=1)
        am_avrg_wtchng_time_co = st.slider("오전 평균 시청 시간 (분)", 0, 180, 40, step=1)
        pm_avrg_wtchng_time_co = st.slider("오후 평균 시청 시간 (분)", 0, 180, 30, step=1)
        evening_avrg_wtchng_time_co = st.slider("저녁 평균 시청 시간 (분)", 0, 180, 20, step=1)

        input_data = [
            avrg_wtchng_co, dawn_avrg_wtchng_time_co, 
            am_avrg_wtchng_time_co, pm_avrg_wtchng_time_co, 
            evening_avrg_wtchng_time_co
        ]

        col1_a, col1_b = st.columns(2)

        # 버튼 추가
        with col1_a:
            if st.button("채팅 추가"):
                # 입력값을 문장으로 변환
                user_message = (
                    f"저는 하루 평균 {avrg_wtchng_co}개의 프로그램을 "
                    f"새벽에 {dawn_avrg_wtchng_time_co}분, 오전에 {am_avrg_wtchng_time_co}분, "
                    f"오후에 {pm_avrg_wtchng_time_co}분, 저녁에 {evening_avrg_wtchng_time_co}분을 시청해요."
                )

                # 예측 수행
                le, scaler, rf = load_models()
                predicted_genre, label = predict_genre(le, scaler, rf, input_data)

                # 챗봇 응답
                chatbot_message = f"예상되는 방송 장르는 '{predicted_genre}'입니다!"
                img_path = os.path.join("img", f"{label}.png")

                # 메시지 묶음 추가
                add_message_bundle(user_message, chatbot_message, img_path)

        with col1_b:
            if st.button("전체 채팅 삭제"):
                clear_chat_history()

    # 오른쪽: 대화 표시
    with col2:
        st.header("챗봇 대화")

        # 모든 채팅 표시
        for bundle in st.session_state["chat_history"]:
            for chat in bundle:
                role, msg = chat["role"], chat["text"]
                if role == "image":
                    st.image(msg, width=200)  # 이미지 표시
                else:
                    st.chat_message(name=role).write(msg)

if __name__ == "__main__":
    main()