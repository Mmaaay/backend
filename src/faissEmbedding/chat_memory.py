import logging
from typing import AsyncGenerator, Dict, List, Tuple
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

When asked about previous questions or conversation history, refer to the retrieved texts.
For Quranic content, use the retrieved texts provided.

User query: {{messages}}

Retrieved History:
{{retrieved_texts}}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="retrieved_texts")  # Added placeholder
    ]
)
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

async def process_chat_stream(
    messages: List[str], 
    retrieved_texts: List[str] = None, 
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """Process chat messages and return AI response as a stream"""
    try:
        formatted_messages = format_messages(messages)
        message_contents = [msg.content for msg in formatted_messages]
        current_question = message_contents[0] if message_contents else ""
        logger.debug(f"Formatted messages: {formatted_messages}")
        logger.debug(f"Message contents: {message_contents}")
        logger.debug(f"Current question: {current_question}")
        
        # Convert retrieved_texts to list of BaseMessage
        if isinstance(retrieved_texts, list):
            retrieved_messages = [HumanMessage(content=msg) for msg in retrieved_texts]
            logger.debug(f"Retrieved_texts as BaseMessages: {retrieved_messages}")
        else:
            retrieved_messages = []
            logger.debug("No retrieved_texts provided or it's not a list.")

        # Setup streaming model
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=2056,
            streaming=True,  # Enable streaming
            timeout=None,
            max_retries=2,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        )

        chain = prompt | model | StrOutputParser()
        
        logger.info("Starting stream for session: %s", session_id)
        async for chunk in chain.astream({
            "messages": message_contents,
            "retrieved_texts": retrieved_messages  # Pass as list of BaseMessage
        }):
            logger.debug(f"Streamed chunk: {chunk}")
            yield chunk

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