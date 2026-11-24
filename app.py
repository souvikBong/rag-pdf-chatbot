import streamlit as st
from ask import ask_question

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Local PDF RAG Chatbot")
st.write("Ask questions about your uploaded PDF")

question = st.text_input(
    "Enter your question"
)

if st.button("Ask"):

    if question.strip():

        with st.spinner("Thinking..."):

            result = ask_question(question)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Retrieved Chunks")

            for idx, doc in enumerate(result["sources"], start=1):
                with st.expander(f"Chunk {idx}"):

                    st.write(doc.page_content)