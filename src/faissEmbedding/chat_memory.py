import asyncio
import logging
import os
import sys
from typing import AsyncGenerator, Dict, List, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import constants
from constants import GEMENI_API_KEY

# Remove ConversationHistory class and its instance as we'll use retrieved messages instead

system_template = """
You are an Arabic Quranic assistant specializing in Tafsir. Your goal is to answer the User's question about the Quran accurately and respectfully, using retrieved Tafsir and Quranic Ayat.

Current Question: {current_question}
Retrieved Context: {context}

Instructions:  
1. Analyze the User's query to understand its intent and focus.  
2. Review the retrieved context if available and incorporate relevant information.
3. Provide clear, direct answers using simple language.
4. Maintain a respectful and precise tone.
"""

# Update prompt template to use simple string formatting
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{question}")  # This will contain the current question
])

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %name%s - %levelname%s - %message%s')
handler.setFormatter(formatter)
logger.addHandler(handler)

async def embed_message(message: str, session_id: str , ):
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

async def process_chat(messages, retrieved_texts: List[dict], session_id: str = None):
    """Process chat messages and return AI response"""
    # Instead of processing directly, use the streaming version
    response_chunks = []
    async for chunk in process_chat_stream(messages, retrieved_texts, session_id):
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
    retrieved_texts: List[str] = None, 
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """Process chat messages and return AI response as a stream with smooth chunking"""
    try:
        current_question = messages if isinstance(messages, str) else messages[0]
        context = "\n".join([str(text) for text in (retrieved_texts or [])][-3:])
        
        logger.info(f"Processing question: {current_question}")
        logger.info(f"With context: {context}")

        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=2056,
            streaming=True,
            timeout=None,
            max_retries=2,
            api_key=GEMENI_API_KEY,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        )

        chain = prompt | model | StrOutputParser()
        
        # Buffer for accumulating partial words/sentences
        text_buffer = ""
        CHUNK_SIZE = 30  # Adjust this value to control streaming smoothness
        
        logger.info("Starting stream for session: %s", session_id)
        async for chunk in chain.astream({
            "current_question": current_question,
            "context": context,
            "question": current_question
        }):
            # Add chunk to buffer
            text_buffer += chunk
            
            # Process complete sentences or when buffer gets too large
            if len(text_buffer) >= CHUNK_SIZE or any(x in text_buffer for x in ['.', '!', '?', '\n']):
                # Split into chunks while preserving word boundaries
                chunks = chunk_text(text_buffer, CHUNK_SIZE)
                
                # Yield all complete chunks except possibly the last one
                for complete_chunk in chunks[:-1]:
                    await asyncio.sleep(0.01)  # Small delay for smoother streaming
                    yield complete_chunk + " "
                
                # Keep any remaining text in the buffer
                text_buffer = chunks[-1] if chunks else ""
        
        # Yield any remaining text in the buffer
        if text_buffer:
            yield text_buffer

        logger.info("Stream completed for session: %s", session_id)

    except Exception as e:
        logger.error("Error in streaming chat: %s", str(e), exc_info=True)
        yield f"Error: {str(e)}"

async def embed_system_response(response: str, session_id: str, question: str):
    """Embeds the system's response with metadata"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'is_response': True,
        'related_question': question
    }
    return await embed_data(response, session_id, metadata)

# ...existing code...