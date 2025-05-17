# StudyBuddi - AI Powered Study Helper

StudyBuddi is an AI-powered application designed to enhance your study experience through various tools leveraging Retrieval-Augmented Generation (RAG) technology.

![StudyBuddi](https://img.shields.io/badge/StudyBuddi-Your%20Perfect%20Study%20Partner-brightgreen)

## Demo Video

<div align="center">
  <a href="https://youtu.be/jlsKxNg11yM?si=1aE7NNvwIgzQWXVD">
    <img src="https://img.youtube.com/vi/jlsKxNg11yM/maxresdefault.jpg" alt="Demo Video" width="100%">
  </a>
</div>

## Features

### 1. Vector RAG - Chat with Your Documents
Upload PDF documents and ask questions about their content. The application processes the documents, creates vector embeddings, and provides accurate answers based on the document content.

### 2. Image Analyzer
Upload images of questions or study materials, and the application will analyze them using OCR technology and provide answers or explanations.

### 3. YouTube Analyzer
Enter a YouTube video URL, and the application will extract the transcript, analyze the content, and provide a concise summary of the video.

### 4. Quiz Generator
Generate interactive quizzes based on your uploaded PDF documents. Test your knowledge with multiple-choice questions and get immediate feedback.

## Technologies Used

- **Streamlit**: For the web application interface
- **LangChain**: For building RAG pipelines
- **Google Gemini AI**: For text generation and embeddings
- **FAISS**: For efficient similarity search and vector storage
- **PyPDF2**: For PDF text extraction
- **YouTube Transcript API**: For extracting YouTube video transcripts
- **OCR.space API**: For image text extraction

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Google API key for Gemini AI

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/studybuddi.git
   cd studybuddi
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

### Running the Application

Run the Streamlit application:
```
streamlit run app.py
```

The application will be available at http://localhost:8501

## Docker Support

You can also run the application using Docker:

1. Build the Docker image:
   ```
   docker build -t studybuddi .
   ```

2. Run the Docker container:
   ```
   docker run -p 8501:8501 studybuddi
   ```

## CI/CD

This project includes GitHub Actions workflows for continuous integration and deployment. When you push to the main branch, the workflow automatically builds and pushes a Docker image to Docker Hub.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
