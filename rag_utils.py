__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv
import pymupdf
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import tiktoken

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
FILE_PATH = "/home/kratos/Documents/agentic-ai-portfolio/Meal_Prep_Plan.pdf"

# Load the .env file containing the OPENAI_API_KEY
load_dotenv()


# Extract text from the input PDF file and create chunks out of it
def extract_and_split_pdf(pdf_path: str = FILE_PATH, stream=None) -> list[Document]:
    if stream:
        doc = pymupdf.open(stream=stream, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=tiktoken.encoding_name_for_model(LLM_MODEL),
        chunk_size=512,
        chunk_overlap=100,
    )

    # Split the text into chunks
    text_chunks = splitter.split_text(all_text)

    documents = [
        Document(page_content=chunk, metadata={"source": FILE_PATH})
        for chunk in text_chunks
    ]

    print(f"Created {len(documents)} chunks")
    return documents


# Create embeddings from the given chunks and store them in a chroma vector database
def create_chroma_db(
    documents: list[Document] = None, embedding_model: str = EMBEDDING_MODEL
) -> Chroma:
    db = Chroma.from_documents(
        documents=documents, embedding=OpenAIEmbeddings(model=embedding_model)
    )
    return db


# Create a RAG chain to enable querying the PDF file using LLMs
def create_rag_chain(db: Chroma, llm_model: str = LLM_MODEL, temperature: float = 0):
    retriever = db.as_retriever(search_type="mmr", k=2, fetch_k=5)

    llm = ChatOpenAI(model=llm_model, temperature=temperature)

    system_prompt = """
    You are a helpful assistant for answering questions from the user. \
    Use the provided context to answer the questions. \
    Do not synthesize any information that's not in the context. 
    If you don't know the answer, say so rather than making something up. \

    Context: {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        messages=[("system", system_prompt), ("human", "{input}")]
    )

    # Sets the context variable in the system prompt from the top-k docs retrieved by retriever
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_docs_chain
    )
    return rag_chain


# To be used for trial run
if __name__ == "__main__":
    documents = extract_and_split_pdf(FILE_PATH)
    db = create_chroma_db(documents, EMBEDDING_MODEL)
    rag_chain = create_rag_chain(db, LLM_MODEL, 0)
    response = rag_chain.invoke({"input": input("You: ").strip()})
    print(response["answer"])
