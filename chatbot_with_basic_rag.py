import streamlit as st
from langchain.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from rag_utils import *


# Generate historic input to include latest 3 QnA pairs in the LLM context
def generate_historic_input(messages: list[BaseMessage]) -> str:
    historic_input = ""
    l = 0 if len(messages) < 6 else -6
    for i, msg in enumerate(messages[l:]):
        if isinstance(msg, HumanMessage):
            historic_input += "\nhuman"
        elif isinstance(msg, AIMessage):
            historic_input += "\nassistant"
        historic_input += msg.content
    historic_input += "\n"
    return historic_input


# Create file uploader
def pdf_file_uploader():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file is not None:
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        if uploaded_file.type == "application/pdf":
            st.write("PDF file uploaded.")
        else:
            print(uploaded_file.type)
            st.write("Only PDF files are allowed.")
            uploaded_file = None
    return uploaded_file


st.title("RAG Chatbot")
st.sidebar.title("PDF Upload")

# Create a list to store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a session state variable to store the RAG chain
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Display the past messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Take input from the chat input box on the webpage
user_input = st.chat_input("Type your message here...")

# Check for uploaded files and update the session RAG chain
uploaded_file = pdf_file_uploader()
if uploaded_file is not None:
    with st.sidebar.status("Processing PDF...", expanded=True) as status:
        st.write("Extracting and splitting text")
        documents = extract_and_split_pdf(stream=uploaded_file.read())
        st.write("Creating vector database")
        db = create_chroma_db(documents=documents)

        st.write("Building RAG chain")
        st.session_state.rag_chain = create_rag_chain(db=db)

        status.update(label="PDF processed!", state="complete")
    st.sidebar.success("Ready to chat")

if user_input:
    # Add current user message to the memory
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Display the current current user message
    with st.chat_message("user"):
        st.write(user_input)

    # Adding the memory from just past 3 QnA pairs along side the document context
    total_input = generate_historic_input(st.session_state.messages) + user_input

    if st.session_state.rag_chain is not None:
        # Invoke the RAG chain with some previous messages and current message as context and input respectively
        response = st.session_state.rag_chain.invoke({"input": total_input})

        # Display the LLM response
        with st.chat_message("assistant"):
            st.write(response["answer"])

        # Add the current LLM response to the memory
        st.session_state.messages.append(AIMessage(content=response["answer"]))

    else:
        # Create a model instance
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Invoke the LLM instance with some previous messages and current message as context and input respectively
        response = llm.invoke(total_input)

        # Display the LLM response
        with st.chat_message("assistant"):
            st.write(response.content)

        # Add the current LLM response to the memory
        st.session_state.messages.append(AIMessage(content=response.content))
