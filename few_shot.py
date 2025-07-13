import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate   # type: ignore
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()


llama ='llama3-70b-8192'

model = ChatGroq(temperature=0, model_name=llama)

if __name__=='__main__':
    
    prompt = ChatPromptTemplate.from_template(
        '''Extract key information from the below text and give output in key-value format.
    Please look at the examples below for reference.

    "question": "The mutual fund generated an 8% return in Q1 2025, with investments in healthcare stocks up by 10%. The expense ratio is 0.60%, and holdings include 25% in real estate.",
    "answer": {{
        "Return in Q1 2025": "8%",
        "Investments in healthcare stocks increased": "10%",
        "Expense ratio": "0.60%",
        "Holdings in real estate": "25%"
    }}

    "question": "In 2024, healthcare spending in the U.S. reached $4.7 trillion, accounting for 18% of GDP. The number of uninsured Americans dropped to 7.2%, while life expectancy rose to 78.8 years. Obesity rates stood at 42%, and the average cost of a hospital stay was $13,262. There were 2.9 physicians per 1,000 people, and the infant mortality rate was 5.4 per 1,000 live births.",
    "answer": {{
        "Healthcare spending in 2024": "$4.7 trillion",
        "Percentage of GDP spent on healthcare": "18%",
        "Uninsured Americans": "7.2%",
        "Life expectancy": "78.8 years",
        "Obesity rate": "42%",
        "Average cost of hospital stay": "$13,262",
        "Physicians per 1,000 people": "2.9",
        "Infant mortality rate": "5.4 per 1,000 live births"
    }}

    {topic}
    '''
    )

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    response = chain.invoke({"topic": '''In 2025, total education spending in the U.S. reached $1.3 trillion, making up 5.1% of GDP.
The national high school graduation rate climbed to 89.4%, while college enrollment stood at 61.2%.
The average student loan debt per borrower reached $37,240, and public school student-teacher ratio was 15.3:1.
Digital learning adoption reached 74%, and average annual tuition for a 4-year public college was $11,860'''})


    print(response)
    