import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq



load_dotenv()

if __name__=='__main__':

    mixtral = 'mixtral-8x7b-32768'
    llama = 'llama3-70b-8192'

    model = ChatGroq(temperature=0, model_name=llama)

    # Initialize memory
    #memory = ConversationBufferMemory(memory_key="chat_history") ## Loads full conversation
    # memory = ConversationBufferWindowMemory(k = 2, memory_key='chat_history') ## memorizes last k question answers

    memory = ConversationSummaryMemory(llm=model, memory_key='chat_history') ## summarizes conversation

    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant that assists humans ONLY on sports. Please answer in 30 words. Take context from {chat_history}",  #""}  8800  10000
            ),
            ("human", "{user_input}"),  #Who is sourav ganguly?
        ]
    )

    # Define the memory transformation
    def memory_to_inputs(inputs):  #{"user_input": "Who is sourav ganguly?"}
        # Retrieve chat history from memory
        memory_data = memory.load_memory_variables({})
        inputs["chat_history"] = memory_data.get("chat_history", "")  #{"user_input": "Who is sourav ganguly?", "chat_hitory": "sourvar --------"}
        return inputs   #{"user_input": "Who is sourav ganguly?", "chat_hitory": ""}
    
    
    chain = RunnableLambda(memory_to_inputs) | template | model | StrOutputParser()

    while True:
        user_input = input("")
        if user_input.lower()=='exit':
            break
        response = chain.invoke({"user_input": user_input})  #{"user_input": "Who is sourav ganguly?"}
        memory.save_context({"input": user_input}, {"output": response})
        print(response)

