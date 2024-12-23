import logging
from typing import Dict, List, Tuple, AsyncGenerator, AsyncIterator
from datetime import datetime

from faissEmbedding.embeddings_manager import (embed_data, embed_system_response,
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

When someone asks "what was my last question", look at the history and find the most recent question (excluding the current one).
For all other questions, provide a direct answer based on your knowledge.

Current conversation history:
{{retrieved_texts}}

Current user question: {{messages}}

Please provide a clear and direct response.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def get_last_question(history_entries: List[dict]) -> str:
    """Return the latest question content from the sorted history list."""
    if not history_entries:
        return "No previous question found."
    
    # Filter for only user questions
    questions = [
        entry for entry in history_entries 
        if entry.get('metadata', {}).get('source') == 'user' 
        and entry.get('metadata', {}).get('is_question')
        and entry.get('content') != "what was my last question"  # Exclude the current question
    ]
    
    return questions[0]["content"] if questions else "No previous question found."

async def process_chat_stream(
    messages, 
    retrieved_texts: str | List[Dict] = None, 
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """Process chat messages and return AI response as a stream"""
    try:
        formatted_messages = format_messages(messages)
        message_contents = [msg.content for msg in formatted_messages]
        current_question = message_contents[0] if message_contents else ""
        
        # Initialize and embed like before
        if retrieved_texts is None or isinstance(retrieved_texts, str):
            retrieved_texts = []
        
        embedding_result = await embed_message(current_question, session_id)
        embedding_status = embedding_result.get("status", "error")
        
        # Format history similar to before but return earlier
        formatted_texts = "No previous conversation history found."
        if embedding_status != "error":
            retrieved_docs = await retrieve_message(current_question, session_id)
            if retrieved_docs:
                history_entries = [
                    doc for doc in retrieved_docs 
                    if isinstance(doc, dict) 
                    and doc.get('metadata', {}).get('timestamp')
                ]
                
                if history_entries:
                    if current_question.lower().strip() == "what was my last question":
                        questions = [
                            doc for doc in history_entries 
                            if doc.get('metadata', {}).get('is_question') 
                            and doc.get('metadata', {}).get('source') == 'user'
                            and doc['content'].lower() != "what was my last question"
                        ]
                        formatted_texts = f"Previous question asked: {questions[0]['content']}" if questions else "No previous questions found."
                    else:
                        history_texts = "\nRecent conversation:\n" + "\n".join([
                            f"- {'Q: ' if doc.get('metadata', {}).get('is_question') else 'A: '}{doc['content']}"
                            for doc in history_entries[:3]
                        ])
                        formatted_texts = history_texts

        # Setup streaming model
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.0,
            max_tokens=2048,
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
        
        print(f"\n[Stream Start] Processing question: {current_question}")
        print(f"[Stream] History context: {formatted_texts[:100]}...")  # Show first 100 chars
        
        response_text = []
        print("\n[Stream] AI Response starting...")
        print("-" * 50)
        async for chunk in chain.astream({
            "messages": message_contents,
            "retrieved_texts": formatted_texts
        }):
            print(f"[Stream Chunk] {chunk}", end="", flush=True)  # Print chunks as they come
            response_text.append(chunk)
            yield chunk
        print("\n" + "-" * 50)
        print("[Stream End] Response complete\n")

        # After streaming is complete, embed the full response
        if embedding_status != "error":
            full_response = "".join(response_text)
            print(f"[Stream] Embedding complete response (length: {len(full_response)})")
            await embed_system_response(full_response, session_id, current_question)

    except Exception as e:
        print(f"\n[Stream Error] {str(e)}")
        logger.error(f"Error in streaming chat: {str(e)}", exc_info=True)
        yield f"Error: {str(e)}"

# Keep the old process_chat for compatibility but make it use the streaming version
async def process_chat(messages, retrieved_texts: str | List[Dict] = None, session_id: str = None):
    response_chunks = []
    async for chunk in process_chat_stream(messages, retrieved_texts, session_id):
        response_chunks.append(chunk)
    
    return {
        "messages": "".join(response_chunks),
        "embedding_status": "success"
    }
