"""
Modern Conversational AI System
A comprehensive conversational AI system with multiple model support,
conversation memory, and modern architecture.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Core AI/ML Libraries
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

# Modern Conversational AI
import openai
from anthropic import Anthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Database
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configuration
from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    session_id = Column(String, index=True)
    message = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_used = Column(String)
    metadata = Column(JSON)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    name = Column(String)
    preferences = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database engine
DATABASE_URL = "sqlite:///./conversational_ai.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

@dataclass
class ConversationTurn:
    user_message: str
    ai_response: str
    timestamp: datetime
    model_used: str
    metadata: Dict[str, Any] = None

@dataclass
class ConversationSession:
    session_id: str
    user_id: str
    turns: List[ConversationTurn]
    created_at: datetime
    updated_at: datetime

class ConversationalAIConfig(BaseSettings):
    """Configuration for the Conversational AI System"""
    
    # Model configurations
    default_model_provider: ModelProvider = ModelProvider.OPENAI
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model settings
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Conversation settings
    max_conversation_history: int = 10
    system_prompt: str = "You are a helpful, knowledgeable, and friendly AI assistant."
    
    # Database settings
    database_url: str = DATABASE_URL
    
    class Config:
        env_file = ".env"

class ModernConversationalAI:
    """Modern Conversational AI System with multiple model support"""
    
    def __init__(self, config: ConversationalAIConfig):
        self.config = config
        self.db_session = SessionLocal()
        self.memory = ConversationBufferWindowMemory(
            k=self.config.max_conversation_history,
            return_messages=True
        )
        
        # Initialize model providers
        self._initialize_providers()
        
        # Load local models if needed
        self.local_model = None
        self.local_tokenizer = None
        
    def _initialize_providers(self):
        """Initialize AI model providers"""
        if self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
            
        if self.config.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.config.anthropic_api_key)
        else:
            self.anthropic_client = None
            
    def load_local_model(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Load a local Hugging Face model"""
        try:
            logger.info(f"Loading local model: {model_name}")
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Set pad token
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
                
            logger.info("Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def generate_response_openai(self, message: str, conversation_history: List[Dict]) -> str:
        """Generate response using OpenAI API"""
        try:
            messages = [{"role": "system", "content": self.config.system_prompt}]
            
            # Add conversation history
            for turn in conversation_history[-self.config.max_conversation_history:]:
                messages.append({"role": "user", "content": turn["user_message"]})
                messages.append({"role": "assistant", "content": turn["ai_response"]})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_response_anthropic(self, message: str, conversation_history: List[Dict]) -> str:
        """Generate response using Anthropic Claude API"""
        try:
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized")
            
            # Build conversation context
            conversation_text = self.config.system_prompt + "\n\n"
            
            for turn in conversation_history[-self.config.max_conversation_history:]:
                conversation_text += f"Human: {turn['user_message']}\n"
                conversation_text += f"Assistant: {turn['ai_response']}\n\n"
            
            conversation_text += f"Human: {message}\nAssistant:"
            
            response = self.anthropic_client.completions.create(
                model="claude-3-sonnet-20240229",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                prompt=conversation_text
            )
            
            return response.completion.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def generate_response_local(self, message: str, conversation_history: List[Dict]) -> str:
        """Generate response using local Hugging Face model"""
        try:
            if not self.local_model or not self.local_tokenizer:
                raise ValueError("Local model not loaded")
            
            # Build conversation context
            conversation_text = ""
            for turn in conversation_history[-self.config.max_conversation_history:]:
                conversation_text += f"User: {turn['user_message']}\nBot: {turn['ai_response']}\n"
            
            conversation_text += f"User: {message}\nBot:"
            
            # Tokenize input
            inputs = self.local_tokenizer.encode(
                conversation_text, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.local_tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Local model error: {e}")
            raise
    
    def generate_response(self, message: str, user_id: str = "default", session_id: str = "default") -> str:
        """Generate AI response using the configured model provider"""
        try:
            # Get conversation history from database
            conversation_history = self._get_conversation_history(user_id, session_id)
            
            # Generate response based on provider
            if self.config.default_model_provider == ModelProvider.OPENAI:
                response = self.generate_response_openai(message, conversation_history)
                model_used = "gpt-3.5-turbo"
            elif self.config.default_model_provider == ModelProvider.ANTHROPIC:
                response = self.generate_response_anthropic(message, conversation_history)
                model_used = "claude-3-sonnet"
            elif self.config.default_model_provider == ModelProvider.HUGGINGFACE:
                response = self.generate_response_local(message, conversation_history)
                model_used = "local-model"
            else:
                raise ValueError(f"Unsupported model provider: {self.config.default_model_provider}")
            
            # Store conversation turn in database
            self._store_conversation_turn(
                user_id, session_id, message, response, model_used
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _get_conversation_history(self, user_id: str, session_id: str) -> List[Dict]:
        """Retrieve conversation history from database"""
        try:
            conversations = self.db_session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.session_id == session_id
            ).order_by(Conversation.timestamp.desc()).limit(self.config.max_conversation_history).all()
            
            history = []
            for conv in reversed(conversations):
                history.append({
                    "user_message": conv.message,
                    "ai_response": conv.response,
                    "timestamp": conv.timestamp,
                    "model_used": conv.model_used
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def _store_conversation_turn(self, user_id: str, session_id: str, message: str, response: str, model_used: str):
        """Store conversation turn in database"""
        try:
            conversation = Conversation(
                user_id=user_id,
                session_id=session_id,
                message=message,
                response=response,
                model_used=model_used,
                metadata={"timestamp": datetime.utcnow().isoformat()}
            )
            
            self.db_session.add(conversation)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            self.db_session.rollback()
    
    def get_conversation_stats(self, user_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a user"""
        try:
            total_conversations = self.db_session.query(Conversation).filter(
                Conversation.user_id == user_id
            ).count()
            
            sessions = self.db_session.query(Conversation.session_id).filter(
                Conversation.user_id == user_id
            ).distinct().count()
            
            return {
                "total_conversations": total_conversations,
                "total_sessions": sessions,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {"error": str(e)}
    
    def clear_conversation_history(self, user_id: str, session_id: str = None):
        """Clear conversation history for a user or specific session"""
        try:
            query = self.db_session.query(Conversation).filter(
                Conversation.user_id == user_id
            )
            
            if session_id:
                query = query.filter(Conversation.session_id == session_id)
            
            query.delete()
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            self.db_session.rollback()

def main():
    """Main function to demonstrate the conversational AI system"""
    # Initialize configuration
    config = ConversationalAIConfig()
    
    # Initialize the AI system
    ai_system = ModernConversationalAI(config)
    
    # Load local model as fallback
    try:
        ai_system.load_local_model()
    except Exception as e:
        logger.warning(f"Could not load local model: {e}")
    
    print("🤖 Modern Conversational AI System")
    print("=" * 50)
    print("Type 'quit' to exit, 'clear' to clear history, 'stats' for statistics")
    print()
    
    user_id = "demo_user"
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'clear':
                ai_system.clear_conversation_history(user_id, session_id)
                print("Conversation history cleared! 🗑️")
                continue
            elif user_input.lower() == 'stats':
                stats = ai_system.get_conversation_stats(user_id)
                print(f"📊 Conversation Statistics: {stats}")
                continue
            elif not user_input:
                continue
            
            # Generate response
            response = ai_system.generate_response(user_input, user_id, session_id)
            print(f"Bot: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()
