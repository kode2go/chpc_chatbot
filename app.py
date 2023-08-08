import streamlit as st
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain import OpenAI
import os
import openai

# Set your API key
openai.api_key = st.secrets["api_secret"]

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 0.5
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.storage_context.persist('index.json')
    return index

def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir='index.json')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

def main():
    st.title("CHPC AI Chatbot")
    input_text = st.text_area("Enter your text", height=200)
    if st.button("Chat"):
        response = chatbot(input_text)
        st.write("Response:")
        st.write(response)

if __name__ == "__main__":
    index = construct_index("docs")
    main()
