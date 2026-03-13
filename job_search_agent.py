from dotenv import load_dotenv
from langchain.tools import tool 
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from tavily import TavilyClient


load_dotenv()

tavily = TavilyClient()

@tool
def search(query:str)->str:
    """
    Search the web for the query.

    Args:
        query: The query to search for.
    Returns:
        The search results.
    """
    print(f"Searching the query: {query} on the web with travily")
    # return "The search results."
    return tavily.search(query=query)




def main():
    
    llm = AzureChatOpenAI()
    tools = [search]
    agent = create_agent(model = llm, tools = tools)
    result = agent.invoke({"messages":HumanMessage(content="how is the weather in Bangalore today?")})
    print(result)

if __name__ == "__main__":
    main()
    