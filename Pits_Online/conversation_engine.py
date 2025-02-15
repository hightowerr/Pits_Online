import os
import sys
import json
import streamlit as st
import traceback
import logging

from openai import OpenAI
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.storage.chat_store import SimpleChatStore
from global_settings import INDEX_STORAGE, CONVERSATION_FILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to console
        logging.FileHandler('conversation_engine.log')  # Log to file
    ]
)
logger = logging.getLogger(__name__)

def load_chat_store():
    try:
        logger.info(f"Attempting to load chat store from {CONVERSATION_FILE}")
        chat_store = SimpleChatStore.from_persist_path(
            CONVERSATION_FILE
        )
        logger.info("Chat store loaded successfully")
    except FileNotFoundError:
        logger.warning("No existing chat store found. Creating a new one.")
        chat_store = SimpleChatStore()
    except Exception as e:
        logger.error(f"Error loading chat store: {e}")
        logger.error(traceback.format_exc())
        chat_store = SimpleChatStore()
    return chat_store

def display_messages(chat_store, container):
    try:
        logger.info("Attempting to display messages")
        with container:
            messages = chat_store.get_messages(key="0")
            logger.info(f"Found {len(messages)} messages")
            for message in messages:
                with st.chat_message(message.role):
                    st.markdown(message.content)
    except Exception as e:
        logger.error(f"Error displaying messages: {e}")
        logger.error(traceback.format_exc())

def initialize_chatbot(user_name, study_subject, 
                       chat_store, container, context):
    try:
        logger.info(f"Initializing chatbot for {user_name}")
        logger.info(f"Study subject: {study_subject}")
        logger.info(f"Context: {context}")

        # Memory setup
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000, 
            chat_store=chat_store, 
            chat_store_key="0"
        )  

        # Index loading
        logger.info(f"Loading index from {INDEX_STORAGE}")
        storage_context = StorageContext.from_defaults(
            persist_dir=INDEX_STORAGE
        )
        
        try:
            index = load_index_from_storage(
                storage_context, index_id="vector"
            )
            logger.info("Index loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            logger.error(traceback.format_exc())
            raise

        # Query engine setup
        study_materials_engine = index.as_query_engine(
            similarity_top_k=3
        )
        study_materials_tool = QueryEngineTool(
            query_engine=study_materials_engine, 
            metadata=ToolMetadata(
                name="study_materials",
                description=(
                    f"Provides official information about "
                    f"{study_subject}. Use a detailed plain "
                    f"text question as input to the tool."
                ),
            )
        )

        # Agent creation
        agent = OpenAIAgent.from_tools(
            tools=[study_materials_tool], 
            memory=memory,
            system_prompt=(
                f"Your name is PITS, a personal tutor. Your "
                f"purpose is to help {user_name} study and "
                f"better understand the topic of: "
                f"{study_subject}. We are now discussing the "
                f"slide with the following content: {context}"
            )
        )
        logger.info("Agent initialized successfully")

        # Display existing messages
        display_messages(chat_store, container)
        
        return agent

    except Exception as e:
        logger.error(f"Critical error in chatbot initialization: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Failed to initialize chatbot: {e}")
        return None

def chat_interface(agent, chat_store, container):  
    try:
        prompt = st.chat_input("Type your question here:")
        if prompt:
            logger.info(f"Received user prompt: {prompt}")
            
            with container:
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                try:
                    response = str(agent.chat(prompt))
                    logger.info("Response generated successfully")
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    logger.error(traceback.format_exc())
                    response = "I'm sorry, I encountered an error processing your request."
                
                # Display agent response
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            # Uncomment to persist chat store (currently disabled)
            # chat_store.persist(CONVERSATION_FILE)
    
    except Exception as e:
        logger.error(f"Error in chat interface: {e}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred: {e}")

# # Optional: Add a main block for standalone testing
# if __name__ == "__main__":
#     logger.info("Running conversation_engine.py directly")
#     try:
#         # Simulate Streamlit environment for testing
#         chat_store = load_chat_store()
#         agent = initialize_chatbot(
#             user_name="Test User", 
#             study_subject="Test Subject", 
#             chat_store=chat_store, 
#             container=st.container(), 
#             context="Test context"
#         )
#         if agent:
#             logger.info("Chatbot initialized successfully")
#         else:
#             logger.error("Failed to initialize chatbot")
#     except Exception as e:
#         logger.error(f"Startup error: {e}")
#         logger.error(traceback.format_exc())
