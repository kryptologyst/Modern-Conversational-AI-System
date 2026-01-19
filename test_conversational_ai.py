"""
Test suite for the Modern Conversational AI System
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

# Import our modules
from conversational_ai import (
    ModernConversationalAI, 
    ConversationalAIConfig, 
    ModelProvider,
    Conversation,
    User
)

class TestConversationalAIConfig:
    """Test the configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ConversationalAIConfig()
        
        assert config.default_model_provider == ModelProvider.OPENAI
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_conversation_history == 10
        assert "helpful" in config.system_prompt.lower()
    
    def test_config_from_env(self):
        """Test configuration from environment variables"""
        with patch.dict(os.environ, {
            'DEFAULT_MODEL_PROVIDER': 'anthropic',
            'MAX_TOKENS': '500',
            'TEMPERATURE': '0.5'
        }):
            config = ConversationalAIConfig()
            assert config.default_model_provider == ModelProvider.ANTHROPIC
            assert config.max_tokens == 500
            assert config.temperature == 0.5

class TestModernConversationalAI:
    """Test the main AI system class"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return ConversationalAIConfig()
    
    @pytest.fixture
    def ai_system(self, config):
        """Create a test AI system"""
        return ModernConversationalAI(config)
    
    def test_initialization(self, ai_system):
        """Test AI system initialization"""
        assert ai_system.config is not None
        assert ai_system.db_session is not None
        assert ai_system.memory is not None
        assert ai_system.local_model is None
        assert ai_system.local_tokenizer is None
    
    def test_provider_initialization(self, ai_system):
        """Test provider initialization"""
        # Test with no API keys
        ai_system._initialize_providers()
        assert ai_system.anthropic_client is None
    
    @patch('openai.ChatCompletion.create')
    def test_openai_response_generation(self, mock_openai, ai_system):
        """Test OpenAI response generation"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_openai.return_value = mock_response
        
        # Test response generation
        message = "Hello"
        history = []
        
        response = ai_system.generate_response_openai(message, history)
        
        assert response == "Hello! How can I help you?"
        mock_openai.assert_called_once()
    
    def test_conversation_storage(self, ai_system):
        """Test conversation storage and retrieval"""
        user_id = "test_user"
        session_id = "test_session"
        message = "Hello"
        response = "Hi there!"
        model_used = "test_model"
        
        # Store conversation
        ai_system._store_conversation_turn(user_id, session_id, message, response, model_used)
        
        # Retrieve conversation history
        history = ai_system._get_conversation_history(user_id, session_id)
        
        assert len(history) == 1
        assert history[0]["user_message"] == message
        assert history[0]["ai_response"] == response
        assert history[0]["model_used"] == model_used
    
    def test_conversation_stats(self, ai_system):
        """Test conversation statistics"""
        user_id = "test_user"
        
        # Add some test conversations
        ai_system._store_conversation_turn(user_id, "session1", "msg1", "resp1", "model1")
        ai_system._store_conversation_turn(user_id, "session1", "msg2", "resp2", "model1")
        ai_system._store_conversation_turn(user_id, "session2", "msg3", "resp3", "model1")
        
        # Get stats
        stats = ai_system.get_conversation_stats(user_id)
        
        assert stats["total_conversations"] == 3
        assert stats["total_sessions"] == 2
        assert stats["user_id"] == user_id
    
    def test_clear_conversation_history(self, ai_system):
        """Test clearing conversation history"""
        user_id = "test_user"
        session_id = "test_session"
        
        # Add some conversations
        ai_system._store_conversation_turn(user_id, session_id, "msg1", "resp1", "model1")
        ai_system._store_conversation_turn(user_id, session_id, "msg2", "resp2", "model1")
        
        # Verify conversations exist
        history = ai_system._get_conversation_history(user_id, session_id)
        assert len(history) == 2
        
        # Clear history
        ai_system.clear_conversation_history(user_id, session_id)
        
        # Verify conversations are cleared
        history = ai_system._get_conversation_history(user_id, session_id)
        assert len(history) == 0

class TestModelProvider:
    """Test the ModelProvider enum"""
    
    def test_model_provider_values(self):
        """Test model provider enum values"""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.HUGGINGFACE.value == "huggingface"
        assert ModelProvider.LOCAL.value == "local"
    
    def test_model_provider_from_string(self):
        """Test creating model provider from string"""
        assert ModelProvider("openai") == ModelProvider.OPENAI
        assert ModelProvider("anthropic") == ModelProvider.ANTHROPIC
        assert ModelProvider("huggingface") == ModelProvider.HUGGINGFACE
        assert ModelProvider("local") == ModelProvider.LOCAL

class TestDatabaseModels:
    """Test database models"""
    
    def test_conversation_model(self):
        """Test Conversation model structure"""
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            message="Hello",
            response="Hi there!",
            model_used="test_model"
        )
        
        assert conversation.user_id == "test_user"
        assert conversation.session_id == "test_session"
        assert conversation.message == "Hello"
        assert conversation.response == "Hi there!"
        assert conversation.model_used == "test_model"
        assert conversation.timestamp is not None
    
    def test_user_model(self):
        """Test User model structure"""
        user = User(
            user_id="test_user",
            name="Test User",
            preferences={"theme": "dark"}
        )
        
        assert user.user_id == "test_user"
        assert user.name == "Test User"
        assert user.preferences == {"theme": "dark"}
        assert user.created_at is not None

# Integration tests
class TestIntegration:
    """Integration tests for the full system"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            temp_db_path = tmp.name
        
        # Update config to use temp database
        config = ConversationalAIConfig()
        config.database_url = f"sqlite:///{temp_db_path}"
        
        yield config
        
        # Cleanup
        os.unlink(temp_db_path)
    
    def test_full_conversation_flow(self, temp_db):
        """Test complete conversation flow"""
        ai_system = ModernConversationalAI(temp_db)
        
        user_id = "integration_test_user"
        session_id = "integration_test_session"
        
        # Mock the response generation to avoid API calls
        with patch.object(ai_system, 'generate_response_openai') as mock_openai:
            mock_openai.return_value = "Hello! How can I help you?"
            
            # Generate response
            response = ai_system.generate_response("Hello", user_id, session_id)
            
            assert response == "Hello! How can I help you?"
            
            # Check that conversation was stored
            history = ai_system._get_conversation_history(user_id, session_id)
            assert len(history) == 1
            assert history[0]["user_message"] == "Hello"
            assert history[0]["ai_response"] == "Hello! How can I help you?"

# Performance tests
class TestPerformance:
    """Performance tests"""
    
    def test_response_time(self):
        """Test response time for local model"""
        config = ConversationalAIConfig()
        config.default_model_provider = ModelProvider.HUGGINGFACE
        
        ai_system = ModernConversationalAI(config)
        
        # This would test actual performance with a real model
        # For now, we'll just test the structure
        assert ai_system.config.default_model_provider == ModelProvider.HUGGINGFACE

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
