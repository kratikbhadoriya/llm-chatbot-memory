# Chatbot with memory

## Description
This is a simple chatbot that uses OpenAI LLMs to repond to user messages. \
It uses Streamlit's session state to maintain the memory for all the reruns of the model in a single user session.

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

## Drawbacks
All the user and assistant messages are added to the context, which will lead to greater token usage.

## Possible Improvements
A conversation summary within a token limit can be maintained as memory to save up on tokens and subsequently the costs.