import asyncio
import logging
from typing import Dict, List, Tuple
from datetime import datetime

from faissEmbedding.embeddings_manager import (embed_data,
                                               retrieve_embedded_data,
                                               state_manager)
from langchain.schema import (AIMessage, BaseMessage, HumanMessage,
                              SystemMessage)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import (ChatGoogleGenerativeAI, HarmBlockThreshold,
                                    HarmCategory)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

# Remove ConversationHistory class and its instance as we'll use retrieved messages instead

system_template = """
You are a helpful arabic assistant specializing in Quranic tafsir. 
Your goal is to answer questions by retrieving relevant tafsir and Quranic ayat.
Be respectful and precise in your responses. Use simple language for better understanding.

When asked about previous questions or conversation history, look at the retrieved texts marked as "history".
For Quranic content, use the retrieved texts marked as "content".

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
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

async def embed_message(message: str, session_id: str):
    """Embeds a message using the embedding manager"""
    return await embed_data(message, session_id)

async def retrieve_message(message: str, session_id: str):
    """Retrieves embedded data for a message"""
    return await retrieve_embedded_data(message, session_id)

def convert_messages_to_string(messages):
    """Convert message objects to string format"""
    formatted_messages = []
    for message in messages:
        if hasattr(message, 'content'):
            formatted_messages.append(message.content)
        elif isinstance(message, dict):
            formatted_messages.append(message.get('content', ''))
        elif isinstance(message, str):
            formatted_messages.append(message)
    return "\n".join(formatted_messages)

def format_messages(message_input) -> list[BaseMessage]:
    """Convert input to proper BaseMessage format"""
    if isinstance(message_input, str):
        return [HumanMessage(content=message_input)]
    
    elif isinstance(message_input, dict):
        content = message_input.get('content', '')
        role = message_input.get('role', 'human')
        if role == 'assistant':
            return [AIMessage(content=content)]
        elif role == 'system':
            return [SystemMessage(content=content)]
        else:
            return [HumanMessage(content=content)]
            
    elif isinstance(message_input, list):
        formatted_messages = []
        for msg in message_input:
            if isinstance(msg, BaseMessage):
                formatted_messages.append(msg)
            elif isinstance(msg, dict):
                content = msg.get('content', '')
                role = msg.get('role', 'human')
                if role == 'assistant':
                    formatted_messages.append(AIMessage(content=content))
                elif role == 'system':
                    formatted_messages.append(SystemMessage(content=content))
                else:
                    formatted_messages.append(HumanMessage(content=content))
            elif isinstance(msg, str):
                formatted_messages.append(HumanMessage(content=msg))
        return formatted_messages
    else:
        raise ValueError("Messages must be a string, dictionary, or list of messages")

async def process_chat(messages, retrieved_texts: List[dict], session_id: str = None, sio=None, sid=None):
    """Process chat messages and return AI response with streaming support"""
    try:
        formatted_messages = format_messages(messages)
        message_contents = [msg.content for msg in formatted_messages]
        current_question = message_contents[0] if message_contents else ""
        
        # Format retrieved_texts to separate history and content
        if isinstance(retrieved_texts, list) and retrieved_texts:
            history_texts = "\nConversation History:\n" + "\n".join([
                f"Q: {doc['content']}" for doc in retrieved_texts 
                if isinstance(doc, dict) and doc.get('type') == 'history'
            ])
            
            content_texts = "\nRetrieved Content:\n" + "\n".join([
                f"Ayat: {doc['content']}" for doc in retrieved_texts 
                if isinstance(doc, dict) and doc.get('type') == 'content'
            ])

            formatted_texts = f"Current Question: {current_question}\n{history_texts}\n{content_texts}"
        else:
            logger.info("No retrieved_texts provided. Using default system context.")
            formatted_texts = f"Current Question: {current_question}\n"

        workflow = StateGraph(state_schema=MessagesState)
        
        parser = StrOutputParser()
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        )

        async def call_model(state: MessagesState):
            chain = prompt | model | parser
            
            response = chain.invoke({
                "messages": message_contents,
                "retrieved_texts": formatted_texts
            })

            # Simulate streaming by sending tokens one by one
            cleaned_response = str(response).strip()
            if isinstance(cleaned_response, str):
                tokens = cleaned_response.split()  # Simple tokenization
                for token in tokens:
                    if sio and sid:
                        asyncio.create_task(sio.emit(
                            "response",
                            {"session_id": session_id, "response": token + " "},
                            to=sid
                        ))
                        await asyncio.sleep(0.5)  # Await sleep asynchronously
                
                # Signal stream end
                if sio and sid:
                    asyncio.create_task(sio.emit(
                        "response",
                        {"session_id": session_id, "response": "", "is_end": True},
                        to=sid
                    ))

            logger.info(f"Generated response for session {session_id}")
            return {"messages": cleaned_response}

        workflow.add_edge(START, "chatbot")
        workflow.add_node("chatbot", call_model)
        workflow.add_edge("chatbot", END)
        
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        config = {
            "thread_id": session_id or str(hash(str(message_contents) + formatted_texts)),
            "checkpoint_ns": "chat_conversation",
            "checkpoint_id": f"chat_{hash(str(message_contents))}"
        }
        
        result = await app.ainvoke({"messages": message_contents}, config=config)
        return str(result.get("messages", ""))
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        raise