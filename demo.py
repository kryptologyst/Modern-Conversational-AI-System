#!/usr/bin/env python3
"""
Demo script for the Modern Conversational AI System
Shows the system capabilities with example conversations.
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_cli_interface():
    """Demo the command line interface"""
    print("🤖 Demo: Command Line Interface")
    print("=" * 50)
    
    try:
        from conversational_ai import ModernConversationalAI, ConversationalAIConfig
        
        # Initialize with demo configuration
        config = ConversationalAIConfig()
        config.default_model_provider = "huggingface"  # Use local model for demo
        
        ai_system = ModernConversationalAI(config)
        
        # Demo conversation
        demo_messages = [
            "Hello! How are you today?",
            "What's the weather like?",
            "Can you help me with a coding problem?",
            "Tell me a joke!"
        ]
        
        user_id = "demo_user"
        session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("Starting demo conversation...")
        print()
        
        for i, message in enumerate(demo_messages, 1):
            print(f"👤 User: {message}")
            
            try:
                # Generate response (this will use local model or show error)
                response = ai_system.generate_response(message, user_id, session_id)
                print(f"🤖 AI: {response}")
            except Exception as e:
                print(f"🤖 AI: I apologize, but I encountered an error: {str(e)}")
                print("   (This is expected if no API keys are configured)")
            
            print()
        
        # Show conversation stats
        stats = ai_system.get_conversation_stats(user_id)
        print(f"📊 Demo Statistics: {stats}")
        
    except Exception as e:
        print(f"❌ Error in CLI demo: {e}")

def demo_api_endpoints():
    """Demo the API endpoints"""
    print("🔌 Demo: API Endpoints")
    print("=" * 50)
    
    print("Available API endpoints:")
    print("• POST /chat - Send message and get AI response")
    print("• GET /conversations/{user_id} - Get conversation history")
    print("• GET /stats/{user_id} - Get user statistics")
    print("• DELETE /conversations/{user_id} - Clear conversation history")
    print("• GET /config - Get system configuration")
    print("• PUT /config - Update system configuration")
    print("• GET /models - Get available models")
    print("• GET /health - Health check")
    print()
    print("📖 Interactive API documentation available at: http://localhost:8000/docs")

def demo_web_interface():
    """Demo the web interface features"""
    print("🌐 Demo: Web Interface Features")
    print("=" * 50)
    
    print("Web interface features:")
    print("• 💬 Real-time chat interface")
    print("• ⚙️ Model configuration sidebar")
    print("• 📊 Analytics dashboard with charts")
    print("• 🕒 Recent activity tracking")
    print("• 💾 Export conversation history")
    print("• 🎛️ Session management controls")
    print("• 📈 Conversation statistics")
    print()
    print("🚀 Start web interface with: python start.py web")
    print("🌍 Access at: http://localhost:8501")

def demo_configuration():
    """Demo configuration options"""
    print("⚙️ Demo: Configuration Options")
    print("=" * 50)
    
    print("Configuration options:")
    print("• 🤖 Model Providers: OpenAI, Anthropic, Hugging Face")
    print("• 🌡️ Temperature: 0.0 - 2.0 (creativity control)")
    print("• 📝 Max Tokens: 50 - 4000 (response length)")
    print("• 💭 Max History: 1 - 50 (conversation memory)")
    print("• 🎯 System Prompt: Customizable AI personality")
    print("• 🔑 API Keys: Secure environment-based storage")
    print()
    print("📝 Configuration file: .env (copy from env.example)")

def demo_deployment():
    """Demo deployment options"""
    print("🚀 Demo: Deployment Options")
    print("=" * 50)
    
    print("Deployment options:")
    print("• 🐳 Docker: Containerized deployment")
    print("• ☁️ Cloud: AWS, GCP, Azure ready")
    print("• 🏠 Local: Development and testing")
    print("• 🔧 Production: Scalable configuration")
    print()
    print("🐳 Docker commands:")
    print("  docker-compose up --build")
    print("  docker build -t conversational-ai .")

def main():
    """Main demo function"""
    print("🎉 Modern Conversational AI System - Demo")
    print("=" * 60)
    print()
    
    demos = [
        ("CLI Interface", demo_cli_interface),
        ("API Endpoints", demo_api_endpoints),
        ("Web Interface", demo_web_interface),
        ("Configuration", demo_configuration),
        ("Deployment", demo_deployment)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"{i}. {name}")
    
    print()
    choice = input("Select demo to run (1-5) or 'all' for all demos: ").strip()
    
    if choice.lower() == 'all':
        for name, demo_func in demos:
            demo_func()
            print()
    else:
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(demos):
                demos[choice_idx][1]()
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")
    
    print()
    print("🎊 Demo complete! Check out the full system:")
    print("• python start.py web    # Web interface")
    print("• python start.py api    # API server")
    print("• python start.py cli    # Command line")
    print("• python start.py test   # Run tests")

if __name__ == "__main__":
    main()
