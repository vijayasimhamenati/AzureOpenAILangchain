from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI


load_dotenv()

# 2. Initialize the Azure Open AI model
# LangChain automatically looks for the AZURE_OPENAI environment variables
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # Must match your Foundry deployment name exactly
    temperature=0.7
)

def main():
    print("Connecting to Azure OpenAI...")
    response = llm.invoke("Hello! Are you successfully connected to LangChain?")
    print("\nResponse from GPT-4o:")
    print("-" * 20)
    print(response.content)


if __name__ == "__main__":
    main()
