import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(
    page_title="eBay Agreement Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    /* base */
    .stApp { background-color: #0f1117; }
    
    /* hide default streamlit chat bubbles */
    .stChatMessage { display: none !important; }
    
    /* main chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 20px 0;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* message row */
    .message-row {
        display: flex;
        align-items: flex-end;
        gap: 10px;
        animation: fadeSlideIn 0.3s ease-out;
    }
    
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    
    /* user row - right aligned */
    .message-row.user {
        flex-direction: row-reverse;
    }
    
    /* avatar circles */
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    .avatar.user { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
    .avatar.bot  { background: linear-gradient(135deg, #0ea5e9, #6366f1); }
    
    /* bubble */
    .bubble {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.6;
        color: #e2e8f0;
    }
    .bubble.user {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-bottom-right-radius: 4px;
        color: white;
    }
    .bubble.bot {
        background-color: #1a1d27;
        border: 1px solid #2d3148;
        border-bottom-left-radius: 4px;
    }
    
    /* typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        animation: fadeSlideIn 0.3s ease-out;
        max-width: 800px;
        margin: 0 auto;
        padding: 0 20px;
    }
    .typing-dots {
        background-color: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        padding: 12px 16px;
        display: flex;
        gap: 5px;
        align-items: center;
    }
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #6366f1;
        animation: bounce 1.2s infinite;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30%            { transform: translateY(-6px); opacity: 1; }
    }

    /* source expander */
    .source-wrapper {
        max-width: 800px;
        margin: -8px auto 0 56px;
    }
    .streamlit-expanderHeader {
        background-color: #1a1d27 !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        color: #6366f1 !important;
        border: 1px solid #2d3148 !important;
    }
    .streamlit-expanderContent {
        background-color: #1a1d27 !important;
        border: 1px solid #2d3148 !important;
        border-radius: 0 0 8px 8px !important;
        font-size: 13px !important;
        color: #8b8fa8 !important;
    }
    blockquote {
        border-left: 3px solid #6366f1;
        padding-left: 12px;
        color: #8b8fa8;
        font-size: 13px;
        margin: 8px 0;
    }

    /* sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d3148;
    }
    
    /* title */
    h1 {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem !important;
        font-weight: 800 !important;
        text-align: center;
    }

    /* chat input */
    .stChatInput > div {
        border-radius: 24px !important;
        background-color: #1a1d27 !important;
        border: 1px solid #2d3148 !important;
    }
    
    /* buttons */
    .stButton button {
        background-color: #6366f1;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton button:hover { background-color: #4f46e5; }

    hr { border-color: #2d3148; }
</style>
""", unsafe_allow_html=True)


# load pipeline
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()


# helpers
def clean_chunk(text):
    for i, char in enumerate(text):
        if i > 0 and char.isupper() and text[i-1] in [' ', '\n']:
            text = text[i:]
            break
    last_period = text.rfind('.')
    if last_period != -1 and last_period > len(text) * 0.5:
        text = text[:last_period + 1]
    return text.strip()

def user_bubble(text):
    return f"""
    <div class="message-row user">
        <div class="avatar user">👤</div>
        <div class="bubble user">{text}</div>
    </div>
    """

def bot_bubble(text):
    return f"""
    <div class="message-row bot">
        <div class="avatar bot">🤖</div>
        <div class="bubble bot">{text}</div>
    </div>
    """

def typing_indicator():
    return """
    <div class="typing-indicator">
        <div class="avatar bot">🤖</div>
        <div class="typing-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    </div>
    """


# sidebar
with st.sidebar:
    st.markdown("## 🤖 eBay Agreement Bot")
    st.markdown("Ask questions about eBay's User Agreement and get accurate, grounded answers.")
    st.divider()
    st.markdown("### ⚙️ System Info")
    st.markdown("**Model:** llama-3.3-70b-versatile")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**Vector DB:** FAISS")
    st.markdown("**Indexed Chunks:** 89")
    st.divider()
    st.markdown("### 💡 Try Asking")
    st.markdown("- What happens if eBay suspends my account?")
    st.markdown("- How does eBay handle disputes?")
    st.markdown("- What are seller fees?")
    st.markdown("- Can eBay change its policies?")
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# main
st.title("🤖 eBay User Agreement Chatbot")
st.caption("<div style='text-align:center; color:#8b8fa8;'>Powered by LLaMA 3.3 70B + RAG • Answers grounded in the official eBay User Agreement</div>", unsafe_allow_html=True)
st.divider()

# init history
if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history with sources inline
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(user_bubble(message["content"]), unsafe_allow_html=True)
    else:
        st.markdown(bot_bubble(message["content"]), unsafe_allow_html=True)
        if "sources" in message and message["sources"]:
            with st.expander("📄 View source passages"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Passage {i+1}**")
                    st.markdown(f"> {clean_chunk(source['chunk'])}")
                    if i < len(message["sources"]) - 1:
                        st.divider()

# chat input
if prompt := st.chat_input("Ask anything about the eBay User Agreement..."):

    # add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # show user bubble immediately
    st.markdown(f'<div class="chat-container">{user_bubble(prompt)}</div>', unsafe_allow_html=True)

    # show typing indicator while generating
    typing_placeholder = st.empty()
    typing_placeholder.markdown(typing_indicator(), unsafe_allow_html=True)

    # stream response
    sources = []
    response_text = ""
    response_placeholder = st.empty()

    for item in pipeline.query_stream(prompt):
        if item["type"] == "sources":
            sources = item["data"]
        elif item["type"] == "token":
            response_text += item["data"]
            # replace typing indicator with streaming text
            typing_placeholder.empty()
            response_placeholder.markdown(
                f'<div class="chat-container">{bot_bubble(response_text + " ▌")}</div>',
                unsafe_allow_html=True
            )

    # final response without cursor
    response_placeholder.markdown(
        f'<div class="chat-container">{bot_bubble(response_text)}</div>',
        unsafe_allow_html=True
    )

    # show sources
    if sources:
        with st.expander("📄 View source passages"):
            for i, source in enumerate(sources):
                st.markdown(f"**Passage {i+1}**")
                st.markdown(f"> {clean_chunk(source['chunk'])}")
                if i < len(sources) - 1:
                    st.divider()

    # save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources
    })

    st.rerun()