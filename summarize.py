from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()

def main():

    information = """Pichai Sundararajan (born June 10, 1972), better known as Sundar Pichai (pronounced: /ˈsʊndɜːr pɪˈtʃeɪ/), is an Indian–American business executive who has been the CEO of Google since 2015 and the CEO of its parent company Alphabet Inc. since 2019.[4]

    Pichai began his career as a materials engineer. Following a short stint at the management consulting firm McKinsey & Co., Pichai joined Google in 2004,[4] where he led the product management and innovation efforts for a suite of Google's client software products, including Google Chrome and ChromeOS, as well as being largely responsible for Google Drive. In addition, he went on to oversee the development of other applications such as Gmail and Google Maps.

    As of February 2026, his net worth is estimated at US$1.6 billion.[5]

    Early life and education
    Pichai was born on June 10, 1972[6][7][8] in Madurai, Tamil Nadu,[9][4][10] to a Tamil Hindu family. His mother, Lakshmi, was a stenographer, and his father, Regunatha Pichai, was an electrical engineer at GEC, the British conglomerate.[11][12]

    Pichai completed schooling in Jawahar Vidyalaya Senior Secondary School[13] in Ashok Nagar, Chennai and completed the Class XII from Vana Vani school at IIT Madras.[14][15] He earned a B.Tech in metallurgical engineering from IIT Kharagpur.[16] He holds an MS from Stanford University in materials science and engineering and an MBA from the Wharton School of the University of Pennsylvania,[17] where he was named a Siebel Scholar and a Palmer Scholar, respectively.[6][18][19]"""

    summary_prompt = """
    Given the following information, summarize it in a few sentences and 2 intresting
    facts about the person.
    {information} """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_prompt
    )

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # Must match your Foundry deployment name exactly
        temperature=0.7
    )

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)






if __name__ == "__main__":
    main()
