# Modern Conversational AI System

A comprehensive conversational AI system with multiple model support, conversation memory, modern web interface, and REST API. Built with the latest AI/ML technologies and best practices.

## Features

### Multiple AI Model Support
- **OpenAI GPT-3.5/4**: Industry-leading language models
- **Anthropic Claude**: Advanced AI with safety focus
- **Hugging Face Models**: Local DialoGPT and other models
- **Easy Model Switching**: Change models on the fly

### Conversation Memory
- **Persistent Storage**: SQLite database for conversation history
- **Session Management**: Track multiple conversation sessions
- **Context Awareness**: Maintains conversation context
- **User Analytics**: Track conversation statistics

### Modern Web Interface
- **Streamlit UI**: Beautiful, interactive web interface
- **Real-time Chat**: Instant AI responses
- **Analytics Dashboard**: Conversation statistics and charts
- **Export Functionality**: Download conversation history

### REST API
- **FastAPI Backend**: High-performance REST API
- **OpenAPI Documentation**: Auto-generated API docs
- **Multiple Endpoints**: Chat, history, stats, configuration
- **CORS Support**: Cross-origin resource sharing

### Advanced Configuration
- **Environment Variables**: Secure configuration management
- **Model Parameters**: Temperature, max tokens, top-p
- **System Prompts**: Customizable AI personality
- **Error Handling**: Robust error management

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- OpenAI API key (optional)
- Anthropic API key (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kryptologyst/Modern-Conversational-AI-System.git
cd Modern-Conversational-AI-System
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp env.example .env
# Edit .env with your API keys
```

4. **Run the system**

**Command Line Interface:**
```bash
python conversational_ai.py
```

**Web Interface:**
```bash
streamlit run web_interface.py
```

**API Server:**
```bash
python api_server.py
```

## Usage

### Command Line Interface

```python
from conversational_ai import ModernConversationalAI, ConversationalAIConfig

# Initialize configuration
config = ConversationalAIConfig()

# Initialize AI system
ai_system = ModernConversationalAI(config)

# Generate response
response = ai_system.generate_response("Hello, how are you?")
print(response)
```

### Web Interface

1. Start the Streamlit app: `streamlit run web_interface.py`
2. Open your browser to `http://localhost:8501`
3. Configure settings in the sidebar
4. Start chatting!

### REST API

**Send a message:**
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, how are you?", "user_id": "user123"}'
```

**Get conversation history:**
```bash
curl "http://localhost:8000/conversations/user123"
```

**Get user statistics:**
```bash
curl "http://localhost:8000/stats/user123"
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `DEFAULT_MODEL_PROVIDER` | Default model provider | openai |
| `MAX_TOKENS` | Maximum response tokens | 1000 |
| `TEMPERATURE` | Response creativity | 0.7 |
| `MAX_CONVERSATION_HISTORY` | History limit | 10 |

### Model Providers

1. **OpenAI** (`openai`)
   - Models: GPT-3.5-turbo, GPT-4
   - Requires: OpenAI API key
   - Best for: General conversation, creative writing

2. **Anthropic** (`anthropic`)
   - Models: Claude-3-sonnet, Claude-3-haiku
   - Requires: Anthropic API key
   - Best for: Analysis, reasoning, safety-focused tasks

3. **Hugging Face** (`huggingface`)
   - Models: DialoGPT-medium, DialoGPT-large
   - Requires: No API key (local)
   - Best for: Privacy, offline usage

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send message and get AI response |
| GET | `/conversations/{user_id}` | Get conversation history |
| GET | `/stats/{user_id}` | Get user statistics |
| DELETE | `/conversations/{user_id}` | Clear conversation history |
| GET | `/config` | Get system configuration |
| PUT | `/config` | Update system configuration |
| GET | `/models` | Get available models |
| GET | `/health` | Health check |

### Interactive API Docs

Visit `http://localhost:8000/docs` for interactive API documentation.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   REST API      │    │   CLI Interface │
│   (Streamlit)   │    │   (FastAPI)     │    │   (Python)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │   Conversational AI Core   │
                    │   (ModernConversationalAI) │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      Model Providers      │
                    │  OpenAI │ Anthropic │ HF  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      Database Layer       │
                    │      (SQLAlchemy)         │
                    └───────────────────────────┘
```

## Testing

### Run Tests
```bash
pytest tests/
```

### Test Coverage
```bash
pytest --cov=conversational_ai tests/
```

## Deployment

### Docker Deployment

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api_server.py"]
```

2. **Build and run**
```bash
docker build -t conversational-ai .
docker run -p 8000:8000 conversational-ai
```

### Production Considerations

- Use environment variables for sensitive data
- Implement proper authentication/authorization
- Set up monitoring and logging
- Use a production database (PostgreSQL, MySQL)
- Configure CORS appropriately
- Implement rate limiting
- Use HTTPS in production

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- Hugging Face for open-source models
- Streamlit for the web framework
- FastAPI for the API framework
- SQLAlchemy for database ORM

## Support

- Create an issue for bug reports
- Start a discussion for questions
- Check the documentation for common issues

## Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Image understanding
- [ ] Custom model fine-tuning
- [ ] Advanced analytics
- [ ] User authentication
- [ ] Conversation templates
- [ ] Plugin system
# Modern-Conversational-AI-System
