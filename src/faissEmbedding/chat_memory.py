from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph,END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.schema import HumanMessage, AIMessage

def create_dynamic_config(session_id):
    return{
        "configurable":{"thread_id":session_id}
    }

# Prepare to call the AI model
parser = StrOutputParser()
config = {}
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

system_template = """
You are a helpful arabic assistant specializing in Quranic tafsir. 
Your goal is to answer questions by retrieving relevant tafsir and Quranic ayat.
Be respectful and precise in your responses. Use simple language for better understanding.
Do not provide any content from your own knowledge. Only provide information from the retrieved texts.
Do not provide harmful or dangerous content.
You can save the conversation into memory for future reference like user names.


User query: {{messages}}
Retrieved texts:
{{retrieved_texts}}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Prepare the workflow for the model
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):

    chain = prompt | model | parser
    response = chain.invoke(state) 
    return {"messages": AIMessage(
            content=response,
            metadata={"role": "ai"},
        )}

workflow.add_edge(START, "chatbot")
workflow.add_node("chatbot", call_model)


memory = MemorySaver()
app = workflow.compile(checkpointer=memory)