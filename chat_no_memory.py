import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


load_dotenv()

if __name__=='__main__':


    llama = 'llama3-70b-8192'

    model = ChatGroq(temperature=0, model_name=llama)
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant that assists humans ONLY on sports. Please answer in 30 words."
            ),
            ("human", "{user_input}"),  #Who is sourav ganguly?
        ]
    )
    
    #{"user_input": user_input}
    chain = RunnablePassthrough() | template | model | StrOutputParser()

    while True:
        user_input = input("")
        if user_input.lower()=='exit':
            break
        response = chain.invoke({"user_input": user_input})  #{"user_input": "Who is sourav ganguly?"}
        
        print(response)

