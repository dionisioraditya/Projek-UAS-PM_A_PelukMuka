import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# Config
# =========================
BASE_MODEL = "unsloth/gemma-2-2b-it"
ADAPTER_REPO = "diordty/gemma2b-lora-pmb-uajy"

st.set_page_config(page_title="Chatbot PMB UAJY (LoRA)", page_icon="🤖", layout="centered")


@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_REPO)
    model.eval()

    return tok, model


def build_prompt(history, user_msg):
    """
    Prompt sederhana instruction-style (aman untuk demo UAS).
    Kamu bisa ganti format sesuai template prompt yang kamu pakai saat training.
    """
    system = (
        "Kamu adalah asisten chatbot PMB UAJY. "
        "Jawab dalam bahasa Indonesia yang jelas, ringkas, dan sopan. "
        "Jika pertanyaan di luar konteks PMB UAJY, katakan tidak yakin dan sarankan cek sumber resmi."
    )

    # Ambil beberapa turn terakhir biar konteks ada tapi tidak kepanjangan
    last_turns = history[-6:]

    convo = ""
    for m in last_turns:
        role = m["role"]
        content = m["content"]
        if role == "user":
            convo += f"User: {content}\n"
        else:
            convo += f"Assistant: {content}\n"

    prompt = (
        f"{system}\n\n"
        f"{convo}"
        f"User: {user_msg}\n"
        f"Assistant:"
    )
    return prompt


@torch.inference_mode()
def generate_answer(tokenizer, model, prompt, max_new_tokens=256, temperature=0.2, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    out = tokenizer.decode(gen[0], skip_special_tokens=True)

    # Ambil hanya bagian setelah "Assistant:" terakhir
    if "Assistant:" in out:
        out = out.split("Assistant:")[-1].strip()

    return out.strip()


# =========================
# UI
# =========================
st.title("🤖 Chatbot PMB UAJY (Gemma 2B + LoRA)")
st.caption("Base model dari Hugging Face + LoRA adapter hasil fine-tuning kamu.")

with st.sidebar:
    st.subheader("⚙️ Pengaturan")
    max_new_tokens = st.slider("Max new tokens", 64, 512, 256, step=16)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)

    st.divider()
    if st.button("🧹 Reset chat"):
        st.session_state.messages = []
        st.rerun()

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load model once
with st.spinner("Loading model + LoRA adapter..."):
    tokenizer, model = load_model()

# Render chat messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_msg = st.chat_input("Tanya seputar PMB UAJY...")

if user_msg:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Build prompt from history
    prompt = build_prompt(st.session_state.messages[:-1], user_msg)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Menjawab..."):
            ans = generate_answer(
                tokenizer, model, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            st.markdown(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})
