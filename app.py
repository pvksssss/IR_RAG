"""Simple NotebookLM — RAG for Learning System.

Streamlit app that lets users upload documents and ask questions about them.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from src.config import config
from src.rag_chain import ingest_document, ask_question, create_llm
from src.vector_store import create_vector_store, delete_collection, reset_vector_store


@st.cache_resource
def get_cached_vector_store(collection_name: str = "documents"):
    return create_vector_store(collection_name)


@st.cache_resource
def get_cached_llm():
    return create_llm()

# Page config
st.set_page_config(
    page_title="Simple NotebookLM",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Simple NotebookLM")
st.caption("Upload tài liệu học tập và đặt câu hỏi — AI sẽ trả lời dựa trên nội dung tài liệu.")

# Initialize loaded docs list
if "loaded_docs" not in st.session_state:
    st.session_state["loaded_docs"] = []

# Sidebar — Document Upload
with st.sidebar:
    st.header("📄 Tài liệu")

    upload_mode = st.radio(
        "Chế độ upload",
        ["Thay thế (xóa tài liệu cũ)", "Thêm vào (giữ tài liệu cũ)"],
        index=0,
    )
    clear_existing = upload_mode.startswith("Thay thế")

    uploaded_file = st.file_uploader(
        "Chọn file (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        help=f"Dung lượng tối đa: {config.max_upload_size_mb}MB",
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > config.max_upload_size_mb:
            st.error(f"File quá lớn: {file_size_mb:.1f}MB. Tối đa: {config.max_upload_size_mb}MB")
        else:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix,
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner("Đang xử lý tài liệu..."):
                try:
                    result = ingest_document(
                        tmp_path,
                        collection_name="documents",
                        clear_existing=clear_existing,
                    )

                    if result["status"] == "success":
                        st.success(
                            f"✅ Đã xử lý **{result['file_name']}**\n\n"
                            f"- {result['num_pages']} trang\n"
                            f"- {result['num_chunks']} đoạn\n"
                            f"- ~{result['avg_chunk_size']} ký tự/đoạn"
                        )
                        get_cached_vector_store.clear()
                        if clear_existing:
                            st.session_state["loaded_docs"] = [result["file_name"]]
                        else:
                            st.session_state["loaded_docs"].append(result["file_name"])
                        st.session_state["doc_loaded"] = True
                        st.session_state["doc_info"] = result
                    else:
                        st.error(result.get("message", "Lỗi không xác định"))

                except Exception as e:
                    st.error(f"Lỗi xử lý tài liệu: {str(e)}")

                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    # Show loaded documents
    if st.session_state["loaded_docs"]:
        st.divider()
        st.subheader("Tài liệu đã tải")
        for doc_name in st.session_state["loaded_docs"]:
            st.caption(f"• {doc_name}")

    # Model info
    st.divider()
    st.header("⚙️ Cấu hình")
    llm_name = config.ollama_llm_model if config.llm_provider == "ollama" else config.gemini_model
    emb_name = config.local_embedding_model if config.embedding_provider == "ollama" else config.gemini_embedding_model
    st.caption(f"**LLM:** {llm_name} ({config.llm_provider})")
    st.caption(f"**Embedding:** {emb_name} ({config.embedding_provider})")
    st.caption(f"**Chunk:** {config.chunk_size} chars (overlap: {config.chunk_overlap})")
    st.caption(f"**Retrieval:** top-{config.k_retrieval}")

    # Clear button
    st.divider()
    if st.button("🗑️ Xóa tài liệu & chat", use_container_width=True):
        if not reset_vector_store():
            delete_collection("documents")
        get_cached_vector_store.clear()
        st.session_state["messages"] = []
        st.session_state["loaded_docs"] = []
        st.session_state["doc_loaded"] = False
        st.session_state["doc_info"] = None
        st.rerun()

# Main chat area
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Nguồn tham khảo"):
                for src in msg["sources"]:
                    st.caption(f"• {src}")

# Chat input
if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm câu trả lời..."):
            try:
                result = ask_question(
                    prompt,
                    collection_name="documents",
                    vector_store=get_cached_vector_store(),
                    llm=get_cached_llm(),
                )
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📎 Nguồn tham khảo"):
                        for src in sources:
                            st.caption(f"• {src}")

                # Save to history
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"❌ Lỗi: {str(e)}"
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                })

# Footer
st.divider()
st.caption(
    "💡 **Mẹo:** Đặt câu hỏi cụ thể về nội dung tài liệu để có câu trả lời chính xác nhất. "
    "Câu hỏi không liên quan sẽ được thông báo là không tìm thấy trong tài liệu."
)
