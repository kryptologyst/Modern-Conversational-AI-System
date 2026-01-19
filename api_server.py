"""
FastAPI Backend for Conversational AI System
Provides REST API endpoints for the conversational AI system.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uvicorn
import logging

# Import our conversational AI system
from conversational_ai import ModernConversationalAI, ConversationalAIConfig, ModelProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Modern Conversational AI API",
    description="A comprehensive REST API for conversational AI with multiple model support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global AI system instance
ai_system = None

def get_ai_system():
    """Get the AI system instance"""
    global ai_system
    if ai_system is None:
        config = ConversationalAIConfig()
        ai_system = ModernConversationalAI(config)
    return ai_system

# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., description="The user's message", min_length=1, max_length=2000)
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    model_provider: Optional[str] = Field(None, description="AI model provider to use")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI's response")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    model_used: str = Field(..., description="Model that generated the response")
    timestamp: datetime = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ConversationHistory(BaseModel):
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    limit: Optional[int] = Field(10, description="Maximum number of conversations to return")

class ConversationStats(BaseModel):
    user_id: str = Field(..., description="User identifier")
    total_conversations: int = Field(..., description="Total number of conversations")
    total_sessions: int = Field(..., description="Total number of sessions")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")

class SystemConfig(BaseModel):
    model_provider: Optional[str] = Field(None, description="Default model provider")
    temperature: Optional[float] = Field(None, description="Response temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens", ge=50, le=4000)
    max_conversation_history: Optional[int] = Field(None, description="Max conversation history", ge=1, le=50)

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Modern Conversational AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "active"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "conversational-ai-api"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_message: ChatMessage,
    ai_system: ModernConversationalAI = Depends(get_ai_system)
):
    """Send a message and get an AI response"""
    try:
        # Set default values
        user_id = chat_message.user_id or "anonymous"
        session_id = chat_message.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Update model provider if specified
        if chat_message.model_provider:
            try:
                ai_system.config.default_model_provider = ModelProvider(chat_message.model_provider)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model provider: {chat_message.model_provider}"
                )
        
        # Generate response
        response = ai_system.generate_response(
            chat_message.message,
            user_id,
            session_id
        )
        
        return ChatResponse(
            response=response,
            user_id=user_id,
            session_id=session_id,
            model_used=ai_system.config.default_model_provider.value,
            timestamp=datetime.utcnow(),
            metadata={
                "temperature": ai_system.config.temperature,
                "max_tokens": ai_system.config.max_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.get("/conversations/{user_id}", response_model=List[Dict[str, Any]])
async def get_conversation_history(
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 10,
    ai_system: ModernConversationalAI = Depends(get_ai_system)
):
    """Get conversation history for a user"""
    try:
        history = ai_system._get_conversation_history(user_id, session_id or "default")
        
        # Format response
        formatted_history = []
        for turn in history[-limit:]:
            formatted_history.append({
                "user_message": turn["user_message"],
                "ai_response": turn["ai_response"],
                "timestamp": turn["timestamp"].isoformat(),
                "model_used": turn["model_used"]
            })
        
        return formatted_history
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}"
        )

@app.get("/stats/{user_id}", response_model=ConversationStats)
async def get_user_stats(
    user_id: str,
    ai_system: ModernConversationalAI = Depends(get_ai_system)
):
    """Get conversation statistics for a user"""
    try:
        stats = ai_system.get_conversation_stats(user_id)
        
        return ConversationStats(
            user_id=user_id,
            total_conversations=stats.get("total_conversations", 0),
            total_sessions=stats.get("total_sessions", 0),
            last_activity=datetime.utcnow()  # This would come from the database in a real implementation
        )
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving user statistics: {str(e)}"
        )

@app.delete("/conversations/{user_id}")
async def clear_conversation_history(
    user_id: str,
    session_id: Optional[str] = None,
    ai_system: ModernConversationalAI = Depends(get_ai_system)
):
    """Clear conversation history for a user or session"""
    try:
        ai_system.clear_conversation_history(user_id, session_id)
        
        return {
            "message": "Conversation history cleared successfully",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing conversation history: {str(e)}"
        )

@app.get("/config", response_model=Dict[str, Any])
async def get_system_config(
    ai_system: ModernConversationalAI = Depends(get_ai_system)
):
    """Get current system configuration"""
    return {
        "model_provider": ai_system.config.default_model_provider.value,
        "temperature": ai_system.config.temperature,
        "max_tokens": ai_system.config.max_tokens,
        "max_conversation_history": ai_system.config.max_conversation_history,
        "system_prompt": ai_system.config.system_prompt
    }

@app.put("/config")
async def update_system_config(
    config: SystemConfig,
    ai_system: ModernConversationalAI = Depends(get_ai_system)
):
    """Update system configuration"""
    try:
        if config.model_provider:
            ai_system.config.default_model_provider = ModelProvider(config.model_provider)
        
        if config.temperature is not None:
            ai_system.config.temperature = config.temperature
        
        if config.max_tokens is not None:
            ai_system.config.max_tokens = config.max_tokens
        
        if config.max_conversation_history is not None:
            ai_system.config.max_conversation_history = config.max_conversation_history
        
        return {
            "message": "Configuration updated successfully",
            "config": {
                "model_provider": ai_system.config.default_model_provider.value,
                "temperature": ai_system.config.temperature,
                "max_tokens": ai_system.config.max_tokens,
                "max_conversation_history": ai_system.config.max_conversation_history
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating configuration: {str(e)}"
        )

@app.get("/models", response_model=List[Dict[str, str]])
async def get_available_models():
    """Get list of available AI models"""
    return [
        {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "description": "OpenAI's GPT-3.5 Turbo model"
        },
        {
            "provider": "openai",
            "model": "gpt-4",
            "description": "OpenAI's GPT-4 model"
        },
        {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "description": "Anthropic's Claude 3 Sonnet model"
        },
        {
            "provider": "anthropic",
            "model": "claude-3-haiku",
            "description": "Anthropic's Claude 3 Haiku model"
        },
        {
            "provider": "huggingface",
            "model": "microsoft/DialoGPT-medium",
            "description": "Microsoft's DialoGPT medium model"
        },
        {
            "provider": "huggingface",
            "model": "microsoft/DialoGPT-large",
            "description": "Microsoft's DialoGPT large model"
        }
    ]

@app.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics(
    ai_system: ModernConversationalAI = Depends(get_ai_system)
):
    """Get system metrics and health information"""
    try:
        # This would typically come from monitoring systems
        return {
            "system_status": "healthy",
            "active_sessions": 1,  # Placeholder
            "total_conversations": 0,  # Placeholder
            "average_response_time": "1.2s",  # Placeholder
            "model_provider": ai_system.config.default_model_provider.value,
            "uptime": "24h",  # Placeholder
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
