from langchain_openai import ChatOpenAI

client = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    temperature=0.9,
)
print(client.invoke("foo"))
