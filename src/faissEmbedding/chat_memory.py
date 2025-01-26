import asyncio
import logging
import os
import sys
from typing import AsyncGenerator, Dict, List, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from db.context import client
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
    history_ai_responses:List[str] = None, 
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """Process chat messages and return AI response as a stream with smooth chunking"""
    try:
        initial_scale_factor = 1  # Initialize the scale factor
        current_question = format_messages(messages) if isinstance(messages, str) else format_messages([messages[0]])
        print(GEMENI_API_KEY)
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=512,
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

        chain = prompt | model
        mongodb_client = client
        checkpointer = AsyncMongoDBSaver(mongodb_client)  
        
        config = {
        "configurable": {
            "thread_id": session_id,
        }}
        
        input_message = {
            "role": "user",
            "content": f"Current Question: {messages} \n Previous Questions: {history_questions} \n Previous Answers: {history_ai_responses}" 
        }
        async def call_model(state: MessagesState):
            chain = prompt | model
            response = await chain.ainvoke({"messages": state["messages"]})
            return {"messages": response}
        
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", END)
        graph = builder.compile(checkpointer=checkpointer)

        
        # Buffer for accumulating partial words/sentences
        text_buffer = ""
        CHUNK_SIZE = 2  # Further reduced chunk size for memory optimization
        
        logger.info("Starting stream for session: %s", session_id)
        print(input_message)
        async for chunk in graph.astream(
            
            {"messages": [input_message]}, config=config, stream_mode='messages'
        ):  
            if not isinstance(chunk, tuple):
                logger.error("Unexpected chunk type: %s", type(chunk))
                continue  # Skip processing this chunk
            message_chunk, metadata = chunk
            text_buffer += message_chunk.content  # Extract content from AIMessageChunk
            if len(text_buffer) >= CHUNK_SIZE or any(x in text_buffer for x in ['.', '!', '?', '\n']):
                chunks = chunk_text(text_buffer, CHUNK_SIZE)
                for complete_chunk in chunks[:-1]:
                    await asyncio.sleep(0.005)  # Reduce sleep time for faster streaming
                    yield complete_chunk + " "
                text_buffer = chunks[-1] if chunks else ""

        # Yield remaining text in the buffer
        if text_buffer:
            yield text_buffer
        
        # Clear buffer
        text_buffer = ""
        chunks = []

        logger.info("Stream completed for session: %s", session_id)

    except Exception as e:
        logger.error("Error in streaming chat: %s", str(e), exc_info=True)
        yield f"Error: {str(e)}"

async def embed_system_response(current_question: str, ai_response: str, session_id: str):
    """Embeds the system's response with metadata"""
    result = await embed_data(current_question, ai_response, session_id)
    return result
    logger.info(f"embed_system_response returned: {result}")
    return result



