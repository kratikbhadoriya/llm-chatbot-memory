from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage
import streamlit as st

# Loading the .env file containing the API key
load_dotenv()

# Create a list to store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the past messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Create a model instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Take input from the chat input box on the webpage
input = st.chat_input("Type your message here...")
if input:
    # Add current user message to the memory
    st.session_state.messages.append(HumanMessage(content=input))

    # Display the current current user message
    with st.chat_message("user"):
        st.write(input)

    # Invoke the LLM instance with all the past messages and current message as context
    response = llm.invoke(st.session_state.messages)

    # Display the LLM response
    with st.chat_message("assistant"):
        st.write(response.content)

    # Add the current LLM response to the memory
    st.session_state.messages.append(AIMessage(content=response.content))
