import asyncio
import gc
import logging
import os
from dotenv import load_dotenv
import sys
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Tuple
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from getpass import getpass
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.schema import (AIMessage, BaseMessage, HumanMessage,
                              SystemMessage)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import (ChatGoogleGenerativeAI, HarmBlockThreshold,
                                    HarmCategory)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from constants import GEMENI_API_KEY, HF_TOKEN
from db.context import client
from faissEmbedding.embeddings_manager import (embed_data,
                                               retrieve_embedded_data,
                                               state_manager)

# Remove ConversationHistory class and its instance as we'll use retrieved messages instead

system_template = """
You are an Arabic Quranic assistant specializing in Tafsir. Your role is to provide accurate, respectful, and concise answers to questions about the Quran, using relevant Tafsir and Quranic Ayat for context.

### Context  
- **Context:** {messages}   

### Instructions  
1. **Understand the Query:**  
   - Analyze the User's question to identify its intent and main focus.  
   - Clarify whether the question relates to a specific Ayah, concept, or theme.  

2. **Review Available Context:**  
   - If retrieved Tafsir or Quranic Ayat are provided, incorporate the most relevant information.  
   - Highlight any connections between the question and the broader Quranic message.  

3. **Craft Your Response:**  
   - Begin with a direct and concise answer.  
   - Use simple and respectful language, avoiding overly complex terms.  
   - Include the relevant Ayah(s) or Tafsir excerpts to support your response.  

4. **Maintain Respectful Tone:**  
   - Address the User with humility and respect.  
   - Ensure that all interpretations align with established Quranic Tafsir principles.  

5. **Focus on Clarity and Relevance:**  
   - Avoid lengthy digressions or unrelated details.  
   - Prioritize clarity and precision in your explanations.  

End your response with an invitation to ask follow-up questions if needed.  

"""

# Update prompt template to use simple string formatting
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder("messages"),

])



# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %name%s - %levelname%s - %message%s')
handler.setFormatter(formatter)
logger.addHandler(handler)

async def embed_message(message: str, user_id: str):
    """Embeds a message using the embedding manager"""
    return await embed_data(message, user_id)

async def retrieve_message(message: str, user_id: str):
    """Retrieves embedded data for a message"""
    return await retrieve_embedded_data(message, user_id)

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
        role = message_input.get('role', 'user')
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
                role = msg.get('role', 'user')
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

async def process_chat(messages, history_questions: List[str], history_ai_responses:List[str], session_id: str = None):
    """Process chat messages and return AI response"""
    # 3. Minimize concurrency by removing extra executor calls
    response_chunks = []
    async for chunk in process_chat_stream(messages, history_questions, history_ai_responses, session_id):
        response_chunks.append(chunk)
    return "".join(response_chunks)

def chunk_text(text: str, chunk_size: int = 10) -> List[str]:
    """Split text into smaller chunks while preserving word boundaries."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_size += len(word) + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

async def process_chat_stream(
    messages: List[str], 
    history_questions: List[str] = None,
    history_ai_responses: List[str] = None, 
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """Memory-optimized chat streaming"""
    try:
        try:
            load_dotenv(dotenv_path=".env")
            
            HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            print("HUGGINGFACEHUB_API_TOKEN", HUGGINGFACEHUB_API_TOKEN)
            
            llm = HuggingFaceEndpoint(
                repo_id="silma-ai/SILMA-9B-Instruct-v1.0",
                task="text-generation",
                do_sample=False,
                repetition_penalty=1.03,
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            )
        except Exception as e:  
            logger.error(f"Failed to initialize HuggingFaceEndpoint: {e}")
            # Handle the error appropriately, e.g., retry or abort

        chat = ChatHuggingFace(llm=llm, verbose=True)
        
        initial_scale_factor = 1  # Initialize the scale factor
        current_question = format_messages(messages) if isinstance(messages, str) else format_messages([messages[0]])
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=256,  # Reduced from 512
            streaming=True,
            timeout=60,  # Increased from 30
            max_retries=1,  # Reduced retries
            api_key=GEMENI_API_KEY,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        )

        
        config = {
        "configurable": {
            "thread_id": session_id,
        }}

        mongodb_client = client
        checkpointer = AsyncMongoDBSaver(mongodb_client)
        data = await checkpointer.aget(config=config)
        last_id = data['channel_values']['messages'][-1].id
        checkpoint_config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_id": last_id
        }}
        
        input_message = {
            "role": "user",
            "content": f"Current Question: {messages} " 
        }
        def get_user_id(config: RunnableConfig) -> str:
            user_id = config["configurable"].get("user_id")
            if user_id is None:
                raise ValueError("User ID needs to be provided to save a memory.")

            return user_id
        
      
        
    


        async def call_model(state: MessagesState):
            chain = prompt | chat
            state["messages"] = []
            checkpoints = checkpointer.alist(config=config, limit=1, before=await checkpointer.aget(config=checkpoint_config))
            checkpoints_list = []
            async for checkpoint in checkpoints:
                checkpoints_list.append(checkpoint)
            for checkpoint in reversed(checkpoints_list):
                if (channel_values := checkpoint.checkpoint.get('channel_values')) and (messages := channel_values.get('messages')):
                    for message in messages: 
                        if isinstance(message, HumanMessage):
                            filtered_message = HumanMessage(
                                content=message.content,
                                additional_kwargs={}
                            )
                            state["messages"].append(filtered_message)
                        if isinstance(message, AIMessage):
                            filtered_message = AIMessage(
                                content=message.content,
                                additional_kwargs={}
                            )
                            state["messages"].append(filtered_message)
                            
            if len(state["messages"]) >= 3:
                state["messages"] = state["messages"][-3:]
                            
            response = await chain.ainvoke({"messages": state["messages"]})
            return {"messages": response}
        
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", END)
        graph = builder.compile(checkpointer=checkpointer)

        
        # Buffer for accumulating the complete response
        text_buffer = ""
        
        logger.info("Starting stream for session: %s", session_id)
        print(input_message)
        # Retrieve embedded data before processing
        retrieved_data = await retrieve_embedded_data(messages, session_id)
        if retrieved_data:
            # Process retrieved data as needed
            pass  # Add any processing logic if necessary

        async for chunk in graph.astream(
            
            {"messages": [input_message]}, config=config, stream_mode='messages' 
        ):  
            if not isinstance(chunk, tuple):
                logger.error("Unexpected chunk type: %s", type(chunk))
                continue  # Skip processing this chunk
            
            message_chunk, metadata = chunk
            text_buffer += message_chunk.content  # Accumulate content

        # Yield the complete text after successful streaming
        yield text_buffer

        # Clear variables periodically
        if len(text_buffer) > 1000:
            text_buffer = text_buffer[-100:]
            gc.collect()
       
        # Yield remaining text in the buffer
        if text_buffer:
            yield text_buffer

        # Clear buffer
        text_buffer = ""
        chunks = []
        temp_ai_temp = None  # Clear any additional temporary variables if present

        logger.info("Stream completed for session: %s", session_id)

    except Exception as e:
        logger.error("Error in streaming chat: %s", str(e), exc_info=True)
        yield f"Error: {str(e)}"
    finally:
        # Cleanup
        await state_manager.clear_cache()
        gc.collect()
        text_buffer = None  # Clear buffer reference
        chunks = None        # Clear chunks reference
        temp_ai_temp = None  # Ensure any temporary variables are cleared

async def embed_system_response(current_question: str, ai_response: str, user_id: str):
    """Embeds the system's response with metadata"""
    result = await embed_data(current_question, ai_response, user_id)
    return result



