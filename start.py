#!/usr/bin/env python3
"""
Startup script for the Modern Conversational AI System
Provides easy access to different components of the system.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import fastapi
        import transformers
        import torch
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def run_cli():
    """Run the command line interface"""
    print("🚀 Starting Conversational AI CLI...")
    try:
        from conversational_ai import main
        main()
    except Exception as e:
        print(f"❌ Error starting CLI: {e}")

def run_web_interface():
    """Run the Streamlit web interface"""
    print("🌐 Starting Web Interface...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "web_interface.py", "--server.port", "8501"
        ])
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

def run_api_server():
    """Run the FastAPI server"""
    print("🔌 Starting API Server...")
    try:
        subprocess.run([
            sys.executable, "api_server.py"
        ])
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def run_tests():
    """Run the test suite"""
    print("🧪 Running Tests...")
    try:
        subprocess.run([
            sys.executable, "-m", "pytest", "test_conversational_ai.py", "-v"
        ])
    except Exception as e:
        print(f"❌ Error running tests: {e}")

def setup_environment():
    """Set up the environment"""
    print("⚙️ Setting up environment...")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        if os.path.exists('env.example'):
            print("📝 Creating .env file from template...")
            with open('env.example', 'r') as src:
                with open('.env', 'w') as dst:
                    dst.write(src.read())
            print("✅ .env file created. Please edit it with your API keys.")
        else:
            print("⚠️ No env.example file found. Creating basic .env...")
            with open('.env', 'w') as f:
                f.write("""# Basic configuration
DEFAULT_MODEL_PROVIDER=openai
MAX_TOKENS=1000
TEMPERATURE=0.7
MAX_CONVERSATION_HISTORY=10
""")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("✅ Environment setup complete!")

def show_status():
    """Show system status"""
    print("📊 System Status")
    print("=" * 50)
    
    # Check files
    files_to_check = [
        'conversational_ai.py',
        'web_interface.py', 
        'api_server.py',
        'requirements.txt',
        'README.md'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
    
    # Check dependencies
    print("\n🔍 Dependencies:")
    check_dependencies()
    
    # Check environment
    print(f"\n🌍 Environment:")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Modern Conversational AI System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py cli          # Run command line interface
  python start.py web           # Run web interface
  python start.py api           # Run API server
  python start.py test          # Run tests
  python start.py setup         # Setup environment
  python start.py status        # Show system status
        """
    )
    
    parser.add_argument(
        'command',
        choices=['cli', 'web', 'api', 'test', 'setup', 'status'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    print("🤖 Modern Conversational AI System")
    print("=" * 50)
    
    if args.command == 'cli':
        run_cli()
    elif args.command == 'web':
        run_web_interface()
    elif args.command == 'api':
        run_api_server()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'setup':
        setup_environment()
    elif args.command == 'status':
        show_status()

if __name__ == "__main__":
    main()
