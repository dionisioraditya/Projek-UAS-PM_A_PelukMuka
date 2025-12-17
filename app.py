import streamlit as st
from llm_hf import LLMClientHF

st.set_page_config(page_title="Chatbot PMB UAJY")

@st.cache_resource
def load_llm():
    return LLMClientHF(
        model_name="placeholder"  # ganti ke repo 
    )

llm = load_llm()

def format_prompt(question, history=""):
    return f"""
Kamu adalah chatbot resmi PMB UAJY.
Jawab pertanyaan dengan jelas, singkat, dan akurat.

{history}
User: {question}
Assistant:
""".strip()

st.title("🎓 Chatbot PMB UAJY")
st.caption("Chatbot berbasis LLM hasil fine-tuning")

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Tanyakan seputar PMB UAJY...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    history_text = ""
    for role, msg in st.session_state.messages[:-1]:
        history_text += f"{role.capitalize()}: {msg}\n"

    prompt = format_prompt(user_input, history_text)

    response = llm.ask(
        prompt,
        temperature=0.0,
        max_tokens=128
    )

    st.session_state.messages.append(("assistant", response))

    with st.chat_message("assistant"):
        st.markdown(response)
