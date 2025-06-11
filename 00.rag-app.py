import os
from dotenv import load_dotenv
from openai import AzureOpenAI


# cmd 명령 함수 생성
def main():
    os.system("cls" if os.name == "nt" else "clear")
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

    # Initialize prompt with system message
    prompt = [
        {
            "role": "system",
            "content": "You are a travel assistant that provides information on travel service available from Margie's Travel.",
        },
    ]

    while True:
        input_text = input("Enter your question (or type 'exit' to quit): ")
        if input_text.lower() == "exit":
            print("Exiting the application.")
            break
        elif input_text.strip() == "":
            print("Please enter a valid question.")
            continue

        # Add user input to the prompt
        prompt.append({"role": "user", "content": input_text})

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
            model=chat_model, messages=prompt, extra_body=rag_params
        )

        completion = response.choices[0].message.content
        print(completion)

        # Add the response to the chat history
        prompt.append({"role": "assistant", "content": completion})


if __name__ == "__main__":
    main()
