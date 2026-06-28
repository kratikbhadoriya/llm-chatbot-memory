# Chatbot with basic RAG

## Description
This is a simple chatbot that uses RAG to repond to user queries on a pdf document. \
It uses Streamlit's session state to maintain the memory for all the reruns of the model/RAG chain in a single user session.

## Execution Steps
1. Create a .env file in the project directory and store your OpenAI API key as shown in the example below.
```code
OPENAI_API_KEY=your_api_key
```
2. Execute the following command in a python shell.
```bash 
streamlit run path/to/script.py
```
3. Open your corresponding Streamlit app in the browser using the following URL. \
http://localhost:8059

4. Ask your questions to the chatbot as you did with the simple chatbot with memory.

5. Upload a .pdf file using the upload file widget, and then ask queries related to the document and its contents.

## Drawbacks
1. The vector databse gets deleted on refreshing the app, as an in memory vector database is used to store the embeddings of the .pdf file.

2. Only the latest 3 QnA pairs are maintained in the context to save up on token consumption.

## Possible Improvements
1. A persistent vector database can be used to persist the embeddings across app restarts.

2. A conversation summary can be maintained to have the context of the whole conversation rather than just last 3 QnA pairs whilst still keeping the token consumption low.