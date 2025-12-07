import os
import streamlit as st

def get_data_files():
    data_files = []
    for dirname, _, filenames in os.walk("data"):
        for filename in filenames:
            data_files.append(os.path.join(filename))
    return data_files


def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.messages = []
