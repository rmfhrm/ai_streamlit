import streamlit as st
from llm import get_ai_messages
from dotenv import load_dotenv

import os

st.title("Streamlit 기본예제")
st.write("소득세에 관련된 모든것을 답변해 드립니다.")

load_dotenv()

# 1. 스트림릿은 자동 재시작되면서 코드를 읽어서 이전의 내용 사라짐. 막으려고 리스트화
if "message_list" not in st.session_state:
    st.session_state.message_list = []

# 2. 기존의 메시지 반복 출력 [사용자]
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

####
# streamlit은 입력하면 처음붙어 끝까지 한번더 쭉 출력 ==> st.session_state -13전째 줄에 있음
####


#print(f"before == {st.session_state.message_list}") -안씀 ?? 왜 ??  # 스트림릿 동작 원리 파악 로그   
if user_question :=st.chat_input(placeholder= '소득세 관련된 궁금한 내용등을 말씀하세요.!'):    # input창을 열어줌
    # 4. 사용자 입력
    with st.chat_message("user"):   #// 내가 입력한 내용 화면에 올라간다.
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
#print(f"after == {st.session_state.message_list}") -안씀 ?? 왜 ??   # 스트림릿 동작 원리 파악 로그

    with st.spinner("답변을 생성하는 중입니다."):
        # 5. AI 출력
        ai_response = get_ai_messages(user_question)    #사용자의 질문이 들어오면 응답값나갈수 있도록 => llm 가기
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)   #llm.py 에서 stream 사용하여 write_stream 써야함
    st.session_state.message_list.append({"role": "AI", "content" : ai_message})