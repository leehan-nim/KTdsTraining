import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st

load_dotenv()  # 환경변수 불러오기

# Get environment variables
openai_endpoint = os.getenv("OPENAI_ENDPOINT")
openai_api_key = os.getenv("OPENAI_API_KEY")
chat_model = os.getenv("CHAT_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")
search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_api_key = os.getenv("SEARCH_API_KEY")
index_name = os.getenv("INDEX_NAME")

# OpenAI client setting
chat_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=openai_endpoint,
    api_key=openai_api_key,
)

st.title("Margie's Travel Assistant")
st.write("Ask your travel-related questions below:")

# streamlit에서 상태를 유지하기 위해서는 st.session_state에 저장이 되어있어야 웹에서 유지됨
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a travel assistant that provides information on travel service available from Margie's Travel.",
        },
    ]

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


# 대화가 이루어질 때마다, 해당 함수를 호출하여 사용할 수 있도록 구현
def get_openai_response(messages):

    # 검색을 위해 azure ai search용 쿼리 작성 필요
    # Additional parameters to apply RAG pattern using the AI Search index
    rag_params = {
        # 1. data sources 세팅
        "data_sources": [
            {
                # 1. rag 타입은 azure search로 갈거고, 인증방식은 key로 하겠다.
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": index_name,
                    "authentication": {
                        "type": "api_key",
                        "key": search_api_key,
                    },
                    # 2. 질의하는 방식(vector 기반)
                    "query_type": "vector",
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": embedding_model,
                    },
                },
            }
        ],
    }

    # Submit the chat request with RAG parameters
    response = chat_client.chat.completions.create(
        model=chat_model, messages=messages, extra_body=rag_params
    )

    completion = response.choices[0].message.content
    return completion


# Handel user input
if user_input := st.chat_input("Enter your question: "):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.spinner("응답을 기다리는 중..."):
        response = get_openai_response(st.session_state.messages)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
