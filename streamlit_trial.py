import pandas as pd
import numpy as np
import streamlit as st
import joblib
import time

title = st.title("테스트")

# 로딩

if "message" not in st.session_state:
    st.session_state["message"] = []

    h1 = st.header("waiting...")
    iter = st.empty()
    bar = st.progress(0)
    for i in range(100):
        iter.text(f"Progress... {i+1} %")
        bar.progress(i+1)
        time.sleep(0.01)
    st.snow()
    time.sleep(0.5)
    iter.text("Done")
    time.sleep(0.5)
    bar.empty()
    iter.empty()
    h1.empty()

# 대화창
# user_input = st.chat_input("물어보세요.")
# if user_input:
#     st.write(f"입력: {user_input}")

# 챗 메시지
# with st.chat_message("user"):
#     st.write("Hello, world!")
#     st.line_chart(np.random.randn(30,3))

# 대화 에코1
# user_input = st.chat_input("물어보세요.")
# if user_input:
#     # input echo
#     st.chat_message(name="user").write(user_input)
#     st.chat_message(name="ai").write(user_input)

# 새로고침 문제 --> st.session_state

for role, msg in st.session_state["message"]:
    st.chat_message(name=role).write(msg)

user_input = st.chat_input("물어보세요.")
if user_input:
    # input echo
    st.chat_message(name="user").write(user_input)
    st.chat_message(name="ai").write(user_input)
    
    # logging
    st.session_state["message"].append(("user", user_input))
    st.session_state["message"].append(("ai", user_input))

