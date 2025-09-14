# ATR Model - Audio Text Response System

A comprehensive AI-powered system that processes audio content and provides intelligent responses using Hugging Face Transformers.

## 🚀 Features

- **Audio Processing**: Upload and transcribe audio files (MP3, WAV, M4A)
- **Intelligent Q&A**: Ask questions about your audio content using Hugging Face QA models
- **Text-to-Speech**: Convert responses back to audio using Piper TTS
- **Real-time Interaction**: Voice-based question and answer system
- **Model Caching**: Efficient model loading and reuse for better performance

## 🏗️ Architecture

### Core Components

1. **Whisper Integration**: Speech-to-text transcription using OpenAI Whisper
2. **Hugging Face QA**: Question-answering using DistilBERT model
3. **Piper TTS**: Text-to-speech conversion for audio responses
4. **FastAPI Backend**: RESTful API for all operations
5. **Web Frontend**: Simple HTML/JS interface for user interaction

### Models Used

- **Whisper**: `whisper-1` (OpenAI) - Audio transcription
- **QA Model**: `distilbert-base-cased-distilled-squad` - Question answering
- **TTS Model**: `en_US-kathleen-low.onnx` (Piper) - Text-to-speech

## 📋 Requirements

### System Requirements
- Python 3.8+
- Windows 10/11 (tested on Windows)
- 4GB+ RAM recommended
- Audio input/output capabilities

### Python Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
transformers==4.35.0
torch==2.1.0
whisper==1.1.10
numpy==1.24.3
pathlib
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ATR-Model
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Piper TTS**:
   - Download Piper executable and model files
   - Place in `piper/` directory
   - Ensure `piper.exe` and `en_US-kathleen-low.onnx` are present

## 🚀 Usage

### Starting the System

1. **Start Backend Server**:
   ```bash
   .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

2. **Start Frontend Server**:
   ```bash
   python -m http.server 3000
   ```

3. **Access the Interface**:
   - Open browser to `http://localhost:3000`
   - Backend API available at `http://127.0.0.1:8000`

### Workflow

1. **Upload Audio**: Select and upload audio files (MP3, WAV, M4A)
2. **Train Model**: Process audio with Whisper and setup QA system
3. **Ask Questions**: Use voice or text to ask questions about the content
4. **Get Responses**: Receive audio responses via TTS

## 📁 Project Structure

```
ATR-Model/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── train.py         # Model training and inference
│   └── state.py         # Training state management
├── training_data/
│   ├── whisper_data/    # Whisper transcripts
│   └── tts_data/        # TTS configuration
├── uploads/             # Uploaded audio files
├── piper/               # Piper TTS files
├── index.html           # Frontend interface
├── styles.css           # Frontend styling
├── script.js            # Frontend JavaScript
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🔧 API Endpoints

### Upload
- **POST** `/upload` - Upload audio files
- **POST** `/train-real` - Train models with uploaded content

### Interaction
- **POST** `/interact` - Ask questions (text or audio)
- **GET** `/progress` - Check training progress

### Management
- **POST** `/fix-training-state` - Manual training state fix

## 🎯 Key Features

### Model Caching
- Models are loaded once at startup
- No reloading during training or inference
- Improved performance and reduced memory usage

### Automatic Data Management
- Clears previous training data on new uploads
- Ensures fresh training for each new file
- Prevents data conflicts

### Clean Answer Generation
- Direct QA model responses
- No hardcoded sentence templates
- Natural, authentic answers

### Error Handling
- Comprehensive error handling throughout
- Graceful fallbacks for failed operations
- Clear error messages for debugging

## 🔍 Troubleshooting

### Common Issues

1. **Upload Conflicts**:
   - System automatically handles file conflicts
   - Old files are removed when uploading new ones

2. **Model Loading Issues**:
   - Ensure all dependencies are installed
   - Check internet connection for model downloads
   - Verify sufficient RAM available

3. **TTS Issues**:
   - Ensure Piper files are in correct location
   - Check file permissions for Piper executable

### Debug Information

- Check server logs for detailed error messages
- Use `/progress` endpoint to verify system status
- Monitor console output for model loading status

## 🚀 Performance Tips

1. **Memory Management**: Close other applications to free RAM
2. **Audio Quality**: Use clear audio files for better transcription
3. **Question Clarity**: Ask specific questions for better answers
4. **Model Caching**: Restart server only when necessary

## 📈 Future Enhancements

- Support for more audio formats
- Additional language models
- Improved TTS voices
- Batch processing capabilities
- Cloud deployment options

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review server logs
3. Create GitHub issue with detailed information

---

**ATR Model** - Making audio content intelligent and interactive! 🎵🤖