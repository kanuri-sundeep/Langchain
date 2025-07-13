import os
from dotenv import load_dotenv


from langchain_core.prompts import ChatPromptTemplate   # type: ignore
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()



llama ='llama3-70b-8192'

model = ChatGroq(temperature=0, model_name=llama)


if __name__=='__main__':
    
    prompt = ChatPromptTemplate.from_template('''Tell me a story about {topic} in {number} words.
                                              
                                              ''')
    
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    response = chain.invoke({"topic":'lion', "number":100})

    print(response)
    