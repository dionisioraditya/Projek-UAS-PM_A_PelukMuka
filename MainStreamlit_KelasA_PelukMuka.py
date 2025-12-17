import streamlit as st
from llm_hf import LLMClientHF

st.set_page_config(page_title="Chatbot PMB UAJY")

@st.cache_resource
def load_llm():
    return LLMClientHF(
        model_name= "diordty/gemma2b-lora-pmb-uajy"
    )

llm = load_llm()

def format_prompt(question, history=""):
    return f"""### Category: PMB_Umum
### Instruction:
Jawablah pertanyaan berikut berdasarkan informasi resmi PMB Universitas Atma Jaya Yogyakarta.

### Input:
{history}
{question}

### Response:
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
    MAX_TURNS = 3
    for role, msg in st.session_state.messages[-2*MAX_TURNS:-1]:
        history_text += f"{role.capitalize()}: {msg}\n"

    prompt = format_prompt(user_input, history_text)

    try:
        response = llm.ask(prompt, temperature=0.0, max_tokens=128)
    except Exception:
        response = "Maaf, sistem sedang mengalami gangguan."

    st.session_state.messages.append(("assistant", response))

    with st.chat_message("assistant"):
        st.markdown(response)
