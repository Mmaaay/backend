import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from langchain.schema import (AIMessage, BaseMessage, HumanMessage,
                              SystemMessage)
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import (ChatGoogleGenerativeAI, HarmBlockThreshold,
                                    HarmCategory)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from constants import GEMENI_API_KEY
from db.context import client
from faissEmbedding.embeddings_manager import (embed_data,
                                               retrieve_embedded_data,
                                               state_manager)

# Remove ConversationHistory class and its instance as we'll use retrieved messages instead

# ...existing code...

combined_system_template = """
You are an Arabic Quranic assistant specializing in Tafsir. Your role is to provide accurate, respectful, and concise answers to questions about the Quran, using releKvant Tafsir and Quranic Ayat for context.

### Context  
    {{messages}}
    
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"    

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

## Additional System Instructions

You are a helpful assistant with advanced long-term memory
capabilities. Powered by a stateless LLM, you must rely on
external memory to store information between conversations.
Utilize the available memory tools to store and retrieve
important details that will help you better attend to the user's
needs and understand their context.

Memory Usage Guidelines:
1. Actively use memory tools (save_core_memory, save_recall_memory)
   to build a comprehensive understanding of the user.
2. Make informed suppositions and extrapolations based on stored
   memories.
3. Regularly reflect on past interactions to identify patterns and
   preferences.
4. Update your mental model of the user with each new piece of
   information.
5. Cross-reference new information with existing memories for
   consistency.
6. Prioritize storing emotional context and personal values
   alongside facts.
7. Use memory to anticipate needs and tailor responses to the
   user's style.
8. Recognize and acknowledge changes in the user's situation or
   perspectives over time.
9. Leverage memories to provide personalized examples and
   analogies.
10. Recall past challenges or successes to inform current
   problem-solving.

## Recall Memories
Recall memories are contextually retrieved based on the current
conversation:
{recall_memories}

# Engage with the user naturally, as a trusted colleague or friend.
# There's no need to explicitly mention your memory capabilities.
# Instead, seamlessly incorporate your understanding of the user
# into your responses. Be attentive to subtle cues and underlying
# emotions. Adapt your communication style to match the user's
# preferences and current emotional state. Use tools to persist
# information you want to retain in the next conversation. If you
# do call tools, all text preceding the tool call is an internal
# message. Respond AFTER calling the tool, once you have
# confirmation that the tool completed successfully.
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", combined_system_template),
        ("placeholder", "{messages}"),
    ]
)
    

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %name%s - %levelname%s - %message%s')
handler.setFormatter(formatter)
logger.addHandler(handler)

async def embed_message(message: str, session_id: str , user_id: str):
    """Embeds a message using the embedding manager"""
    return await embed_data(message, session_id , user_id)

async def retrieve_message(message: str , user_id ,k_num=5):
    """Retrieves embedded data for a message"""
    return await retrieve_embedded_data(message ,user_id ,k_num)

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

async def process_chat(messages, history_questions: List[str], history_ai_responses:List[str] ,session_id: str = None):
    """Process chat messages and return AI response"""
    # Instead of processing directly, use the streaming version
    response_chunks = []
    async for chunk in process_chat_stream(messages, history_questions,history_ai_responses, session_id):
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
    session_id: str = None,
    user_id: str = None
) -> AsyncGenerator[str, None]:
    """Process chat messages and return AI response as a stream with smooth chunking"""
    try:

        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=1024,
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
        
        mongodb_client = client
        checkpointer = AsyncMongoDBSaver(mongodb_client)
            
            
        config = {
        "configurable": {
            "user_id": user_id,
            "thread_id": session_id,
        }}
        chain = prompt | model | StrOutputParser()
        
        ##for testing purposes
        @tool
        async def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
            """Search for relevant memories based on the current conversation context."""
            user_id = config['configurable'].get('user_id')
            
            thread_id = config['configurable'].get('thread_id')
            return await retrieve_embedded_data(query, user_id, k_num=3)
        
        @tool
        async def save_recall_memory(memory: str, config: RunnableConfig) -> str:
            """Save memory to vectorstore for later semantic retrieval."""
            thread_id = config["thread_id"]
            user_id = config["user_id"]
            return await embed_data(memory, user_id)
        
        
        tools = [save_recall_memory, search_recall_memories]
        
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        class State(MessagesState):
    # add memories that will be retrieved based on the conversation context
            recall_memories: List[str]
        model_with_tools = model.bind_tools(tools)   
        
        async def agent(state: State) -> State:
            """Process the current state and generate a response using the LLM.

            Args:
                state (schemas.State): The current state of the conversation.

            Returns:
                schemas.State: The updated state with the agent's response.
            """
            bound = prompt | model_with_tools
            print("state", state)
            recall_str = (
                "<recall_memory>\n" + "\n".join([f"User: {mem['user_question']}\nAI: {mem['ai_response']}" for mem in state["recall_memories"]]) + "\n</recall_memory>"
            )
            prediction = bound.invoke(
                {
                    "messages": state["messages"],
                    "recall_memories": recall_str,
                }
            )
            return {
                "messages": [prediction],
            }


        async def load_memories(state: State, config: RunnableConfig) -> State:
            """Load memories for the current conversation.

            Args:
                state (schemas.State): The current state of the conversation.
                config (RunnableConfig): The runtime configuration for the agent.

            Returns:
                State: The updated state with loaded memories.
            """
            convo_str = get_buffer_string(state["messages"])
            convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
            recall_memories = await search_recall_memories.ainvoke(convo_str, config)
            return {
                "recall_memories": recall_memories,
            }


        def route_tools(state: State):
            """Determine whether to use tools or end the conversation based on the last message.

            Args:
                state (Schemas.State): The current state of the conversation.

            Returns:
                Literal["tools", "__end__"]: The next step in the graph.
            """
            msg = state["messages"][-1]
            if msg.tool_calls:
                return "tools"

            return END
        
        builder = StateGraph(State)
        builder.add_node(load_memories)
        builder.add_node(agent)
        builder.add_node("tools", ToolNode(tools))

        # Add edges to the graph
        builder.add_edge(START, "load_memories")
        builder.add_edge("load_memories", "agent")
        builder.add_conditional_edges("agent", route_tools, ["tools", END])
        builder.add_edge("tools", "agent")

        # Compile the graph
        graph = builder.compile(checkpointer=checkpointer)
       
        input_message = {
            "role": "user",
            "content": f"Current Question: {messages}"
        }

        # Buffer for accumulating partial words/sentences
        text_buffer = ""
        CHUNK_SIZE = 30  # Adjust this value to control streaming smoothness
        
        logger.info("Starting stream for session: %s", session_id)
        
        async for chunk in graph.astream(
            {"messages": [input_message]}, config=config
        ):  
            if not isinstance(chunk, dict):
                logger.error("Unexpected chunk type: %s", type(chunk))
                continue  # Skip processing this chunk

            text_buffer += chunk.get("messages", "")  # Ensure key exists

            if len(text_buffer) >= CHUNK_SIZE or any(x in text_buffer for x in ['.', '!', '?', '\n']):
                chunks = chunk_text(text_buffer, CHUNK_SIZE)
                for complete_chunk in chunks[:-1]:
                    await asyncio.sleep(0.01)
                    yield complete_chunk + " "
                text_buffer = chunks[-1] if chunks else ""

        
        if text_buffer:
            yield text_buffer
    except Exception as e:
        logger.error("Error in streaming chat: %s", str(e), exc_info=True)
        yield f"Error: {str(e)}"

async def embed_system_response(current_question: str, ai_response: str, session_id: str):
    """Embeds the system's response with metadata"""
  
    return await embed_data(current_question, ai_response, session_id)

