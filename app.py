import streamlit as st
from agents.search_agent import search_stream
from agents.rag_agent import build_vector_store, ask_stream, get_retrieved_chunks

# ---- Page Config ----
st.set_page_config(page_title="My First Chatbot", page_icon="🤖", layout="centered")

# ---- Modern AI Theme ----
st.markdown("""
<style>

/* App Background */
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b,#020617);
    color: #e2e8f0;
}

/* Title */
h1 {
    text-align: center;
    font-weight: 700;
    font-size: 2rem;
    background: linear-gradient(90deg, #cddce2, #7e81f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.7);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Chat message bubble */
.stChatMessage {
    background: rgba(23, 30, 54, 0.37);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.05);
    padding: 14px;
    margin-bottom: 10px;
}

/* Chat input */
textarea {
    background: rgba(2,6,23,0.8) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg,#6366f1,#06b6d4);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton button:hover {
    background: linear-gradient(135deg,#4f46e5,#0891b2);
}

/* Agent badges */
.agent-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 14px;
    font-size: 0.75em;
    font-weight: 600;
}

.badge-search {
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
}

.badge-rag {
    background: rgba(16,185,129,0.15);
    color: #6ee7b7;
}

/* Code blocks */
pre {
    background: #020617 !important;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08);
}

/* Chunk meta */
.chunk-meta {
    font-size: 0.8em;
    color: #94a3b8;
}

.relevance-high { color: #22c55e; }
.relevance-mid { color: #facc15; }
.relevance-low { color: #ef4444; }

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.3);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("My first AI Chatbot")


def render_chunks(chunks):
    """Renders retrieved chunks with relevance scores and metadata."""
    for i, chunk in enumerate(chunks, 1):
        relevance = chunk.get("relevance", 0)
        score = chunk.get("score", 0)
        source = chunk.get("source", "Unknown")
        page = chunk.get("page")

        # Color based on relevance
        if relevance >= 60:
            color_class = "relevance-high"
        elif relevance >= 40:
            color_class = "relevance-mid"
        else:
            color_class = "relevance-low"

        # Header with chunk number, relevance, and metadata
        page_info = f" | Page {page + 1}" if page is not None else ""
        st.markdown(
            f'<div class="chunk-header">'
            f'<strong>Chunk {i}</strong>'
            f'<span class="chunk-meta">'
            f'<span class="{color_class}">Relevance: {relevance}%</span>'
            f' | Distance: {score:.3f}'
            f'{page_info}'
            f' | Source: {source}'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        st.code(chunk["content"], language=None)
        if i < len(chunks):
            st.divider()


# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")

    st.markdown('<p class="sidebar-info">Powered by GPT-OSS-120B via Groq</p>', unsafe_allow_html=True)

    agent_choice = st.radio("Choose Agent:", ["Search", "RAG"], index=0)

    st.divider()

    # Show file uploader only when RAG is selected
    if agent_choice == "RAG":
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files", type=["pdf", "txt"], accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        st.session_state.vector_store = build_vector_store(uploaded_files)
                        st.success(f"Processed {len(uploaded_files)} file(s)!")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")

    # Show which agent is active
    if agent_choice == "Search":
        st.info("Web search is active. Ask anything about current events.")
    else:
        if "vector_store" in st.session_state:
            st.success("Documents loaded. Ask questions about them.")
        else:
            st.warning("Upload and process PDF/TXT files to start.")

    st.divider()

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.search_messages = []
        st.session_state.rag_messages = []
        st.rerun()

    # Export chat button
    current_key = "search_messages" if agent_choice == "Search" else "rag_messages"
    if current_key in st.session_state and st.session_state[current_key]:
        chat_export = ""
        for msg in st.session_state[current_key]:
            role = "You" if msg["role"] == "user" else f"Assistant ({msg.get('agent', agent_choice)})"
            chat_export += f"**{role}:**\n{msg['content']}\n\n---\n\n"
        st.download_button(
            "Export Chat",
            data=chat_export,
            file_name=f"chat_{agent_choice.lower()}.md",
            mime="text/markdown",
        )

# ---- Initialize Chat History (per agent) ----
if "search_messages" not in st.session_state:
    st.session_state.search_messages = []
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

# Pick the right history based on agent
messages = st.session_state.search_messages if agent_choice == "Search" else st.session_state.rag_messages

# ---- Display Chat History ----
for message in messages:
    with st.chat_message(message["role"]):
        # Show agent badge on assistant messages
        if message["role"] == "assistant":
            badge_class = "badge-search" if message.get("agent") == "Search" else "badge-rag"
            badge_label = message.get("agent", agent_choice)
            st.markdown(
                f'<span class="agent-badge {badge_class}">{badge_label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(message["content"])

        # Show retrieved chunks if available (RAG)
        if message.get("chunks"):
            with st.expander("Retrieved Context"):
                render_chunks(message["chunks"])

# ---- Handle User Input ----
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate streamed response
    with st.chat_message("assistant"):
        # Show agent badge
        badge_class = "badge-search" if agent_choice == "Search" else "badge-rag"
        st.markdown(
            f'<span class="agent-badge {badge_class}">{agent_choice}</span>',
            unsafe_allow_html=True,
        )

        try:
            if agent_choice == "Search":
                response = st.write_stream(search_stream(prompt, messages[:-1]))
                messages.append({"role": "assistant", "content": response, "agent": "Search"})

            elif agent_choice == "RAG":
                if "vector_store" not in st.session_state:
                    response = "Please upload and process documents first using the sidebar."
                    st.markdown(response)
                    messages.append({"role": "assistant", "content": response, "agent": "RAG"})
                else:
                    # Get retrieved chunks for display
                    chunks = get_retrieved_chunks(prompt, st.session_state.vector_store)

                    response = st.write_stream(
                        ask_stream(prompt, st.session_state.vector_store, messages[:-1])
                    )

                    # Show retrieved chunks in expander
                    with st.expander("Retrieved Context"):
                        render_chunks(chunks)

                    messages.append({
                        "role": "assistant",
                        "content": response,
                        "agent": "RAG",
                        "chunks": chunks,
                    })

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.error("Rate limit reached. Please wait a moment and try again.")
            elif "api_key" in error_msg.lower() or "401" in error_msg:
                st.error("Invalid API key. Please check your GROQ_API_KEY in the .env file.")
            else:
                st.error(f"Something went wrong: {error_msg}")
