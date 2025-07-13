import os
from langchain_core.prompts import ChatPromptTemplate   # type: ignore
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()



llama ='llama3-70b-8192'
llama = 'llama-3.1-8b-instant'

model = ChatGroq(temperature=0, model_name=llama)


if __name__=='__main__':
    
    prompt = ChatPromptTemplate.from_template('''Sarah has a rectangular garden. She wants to plant flowers in a pattern where each row has 3 more flowers than the previous row. If she starts with 8 flowers in the first row and plants 6 rows total, then decides to add a border around the entire garden using 1 flower per foot of perimeter, how many flowers does she need if the garden is 12 feet wide and 15 feet long?
                                              
                                              ''')

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    response = chain.invoke({})

    print(response)
    