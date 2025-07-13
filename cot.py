import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()


mixtral = 'mixtral-8x7b-32768'
llama = 'llama3-70b-8192'

model = ChatGroq(temperature=0, model_name=llama)

# Create the prompt
prompt = ChatPromptTemplate.from_template(
        '''
    Please answer below question
    {question}


Let's solve this step by step:
1) First, let's identify the initial number of apples
2) Then, track how the number changes with each event
3) Finally, calculate the final number

Please show your reasoning for each step, then provide the final answer.

    '''
    )

# Create the chain
chain = prompt | model | StrOutputParser()

# Example question
question = "I have 12 apples, i gave 3 apples to my friend and i bought 5 more apples, i gave 6 apples to my brother, i again bough an apple. how many apples i'm left with"

# Run the chain
response = chain.invoke({'question': question})
print(response)