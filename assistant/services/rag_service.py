# assistant/services/rag_service.py
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # Load OPENAI_API_KEY etc.

EMBEDDING_MODEL = OpenAIEmbeddings()
VECTOR_DB_PATH = Path("chroma_db")


# ---- Simple text splitter (no langchain.text_splitter dependency) ----
def simple_split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200):
    """
    Very simple text splitter:
    - Splits the text into chunks of length `chunk_size`
    - Each chunk overlaps the previous chunk by `chunk_overlap` characters
    """
    chunks = []
    start = 0
    length = len(text)

    # Safety: clamp overlap
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - chunk_overlap

    return chunks


class SimpleRAGService:
    def __init__(self):
        # Create / load persistent Chroma DB
        self.vectorstore = Chroma(
            collection_name="assistant_rag",
            embedding_function=EMBEDDING_MODEL,
            persist_directory=str(VECTOR_DB_PATH),
        )
        # LLM used to answer questions
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # Add raw texts directly (optional helper)
    def add_documents(self, texts):
        docs = [Document(page_content=t) for t in texts]
        self.vectorstore.add_documents(docs)

    # Add documents from a single file path
    def add_documents_from_file(self, file_path: str):
        text = Path(file_path).read_text(encoding="utf-8")
        chunks = simple_split_text(text, chunk_size=800, chunk_overlap=200)
        docs = [Document(page_content=c) for c in chunks]
        self.vectorstore.add_documents(docs)

    # Main Q&A API
    def ask(self, question: str) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant. Use the context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
        )

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )

        resp = chain.invoke(question)
        return resp.content


# Global instance used by Django views
rag_service = SimpleRAGService()
