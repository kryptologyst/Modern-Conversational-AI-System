"""
Modern Web Interface for Conversational AI System
Built with Streamlit for a beautiful, interactive user experience.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import uuid
from typing import List, Dict, Any

# Import our conversational AI system
from conversational_ai import ModernConversationalAI, ConversationalAIConfig, ModelProvider

# Page configuration
st.set_page_config(
    page_title="🤖 Modern Conversational AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .stats-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_ai_system():
    """Initialize the AI system with caching"""
    config = ConversationalAIConfig()
    return ModernConversationalAI(config)

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = initialize_ai_system()

def display_chat_message(message: str, is_user: bool = True):
    """Display a chat message with appropriate styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)

def display_conversation_history():
    """Display the conversation history"""
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation History")
        
        for turn in st.session_state.conversation_history:
            display_chat_message(turn['user_message'], is_user=True)
            display_chat_message(turn['ai_response'], is_user=False)
            st.markdown(f"*{turn['timestamp']}*")
            st.markdown("---")

def get_conversation_stats():
    """Get and display conversation statistics"""
    try:
        stats = st.session_state.ai_system.get_conversation_stats(st.session_state.user_id)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Total Conversations",
                value=stats.get('total_conversations', 0)
            )
        
        with col2:
            st.metric(
                label="🔄 Total Sessions",
                value=stats.get('total_sessions', 0)
            )
        
        with col3:
            st.metric(
                label="👤 User ID",
                value=st.session_state.user_id[:8] + "..."
            )
        
        return stats
        
    except Exception as e:
        st.error(f"Error getting stats: {e}")
        return {}

def create_conversation_chart():
    """Create a chart showing conversation activity"""
    try:
        # This would typically come from the database
        # For demo purposes, we'll create sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
        conversations = [5, 8, 12, 6, 15, 9, 11]  # Sample data
        
        fig = px.line(
            x=dates,
            y=conversations,
            title="📈 Conversation Activity (Last 7 Days)",
            labels={'x': 'Date', 'y': 'Number of Conversations'}
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Modern Conversational AI System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model selection
        model_provider = st.selectbox(
            "🤖 AI Model Provider",
            options=[ModelProvider.OPENAI.value, ModelProvider.ANTHROPIC.value, ModelProvider.HUGGINGFACE.value],
            index=0
        )
        
        # Update model provider
        if model_provider != st.session_state.ai_system.config.default_model_provider.value:
            st.session_state.ai_system.config.default_model_provider = ModelProvider(model_provider)
        
        # Temperature setting
        temperature = st.slider(
            "🌡️ Response Creativity",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make responses more creative and random"
        )
        st.session_state.ai_system.config.temperature = temperature
        
        # Max tokens
        max_tokens = st.slider(
            "📝 Max Response Length",
            min_value=50,
            max_value=2000,
            value=1000,
            step=50,
            help="Maximum number of tokens in the response"
        )
        st.session_state.ai_system.config.max_tokens = max_tokens
        
        st.markdown("---")
        
        # Statistics
        st.markdown("## 📊 Statistics")
        stats = get_conversation_stats()
        
        st.markdown("---")
        
        # Controls
        st.markdown("## 🎛️ Controls")
        
        if st.button("🗑️ Clear History", help="Clear conversation history"):
            st.session_state.ai_system.clear_conversation_history(
                st.session_state.user_id, 
                st.session_state.session_id
            )
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
            st.rerun()
        
        if st.button("🔄 New Session", help="Start a new conversation session"):
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.conversation_history = []
            st.success("New session started!")
            st.rerun()
        
        # Export conversation
        if st.session_state.conversation_history:
            conversation_json = json.dumps(st.session_state.conversation_history, indent=2, default=str)
            st.download_button(
                label="💾 Export Conversation",
                data=conversation_json,
                file_name=f"conversation_{st.session_state.session_id}.json",
                mime="application/json"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Chat Interface")
        
        # Chat input
        user_input = st.text_input(
            "Type your message here...",
            placeholder="Ask me anything!",
            key="chat_input"
        )
        
        # Send button
        if st.button("🚀 Send", type="primary") or user_input:
            if user_input.strip():
                with st.spinner("🤖 AI is thinking..."):
                    try:
                        # Generate AI response
                        ai_response = st.session_state.ai_system.generate_response(
                            user_input,
                            st.session_state.user_id,
                            st.session_state.session_id
                        )
                        
                        # Store in session state
                        turn = {
                            'user_message': user_input,
                            'ai_response': ai_response,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'model_used': st.session_state.ai_system.config.default_model_provider.value
                        }
                        
                        st.session_state.conversation_history.append(turn)
                        
                        # Clear input
                        st.session_state.chat_input = ""
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("---")
            display_conversation_history()
        else:
            st.info("👋 Start a conversation by typing a message above!")
    
    with col2:
        st.markdown("### 📈 Analytics")
        
        # Conversation activity chart
        create_conversation_chart()
        
        # Recent activity
        st.markdown("### 🕒 Recent Activity")
        if st.session_state.conversation_history:
            recent_turns = st.session_state.conversation_history[-3:]
            for turn in reversed(recent_turns):
                st.markdown(f"""
                <div class="stats-card">
                    <strong>{turn['timestamp']}</strong><br>
                    <small>Model: {turn['model_used']}</small><br>
                    <small>Message: {turn['user_message'][:50]}...</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent activity")
        
        # Model information
        st.markdown("### 🤖 Model Info")
        st.markdown(f"""
        <div class="stats-card">
            <strong>Provider:</strong> {st.session_state.ai_system.config.default_model_provider.value}<br>
            <strong>Temperature:</strong> {st.session_state.ai_system.config.temperature}<br>
            <strong>Max Tokens:</strong> {st.session_state.ai_system.config.max_tokens}<br>
            <strong>History Limit:</strong> {st.session_state.ai_system.config.max_conversation_history}
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🤖 Modern Conversational AI System | Built with Streamlit | 
        <a href="https://github.com" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
