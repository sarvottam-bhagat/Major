import streamlit as st
from PyPDF2 import PdfReader
import random
from PIL import Image
import io
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

st.set_page_config(
    page_title="SIVI AI Buddi",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
def get_conversational_chain():
    prompt_template = """
    As an AI study assistant, your role is to thoroughly analyze the provided document and provide accurate, detailed answers to questions based on its content.
    
    Assume the role of a knowledgeable tutor who can explain complex topics in a clear, concise, and student-friendly manner.

    You should be able to:
    1. **Understand the core concepts**: Identify and explain the key ideas, theories, formulas, or processes presented in the material.
    2. **Provide relevant examples**: Offer specific examples from the document to illustrate the concepts being discussed.
    3. **Clarify doubts**: Address any specific areas of confusion the student might have regarding the subject matter.
    4. **Offer study tips**: Suggest effective strategies for understanding and memorizing the material, if applicable.
    5. **Summarize sections**: Provide concise summaries of different parts of the document to aid in quick revision.

    Make sure your answers are directly based on the information in the document and avoid assumptions or external information. Be as thorough as possible, aiming to enhance the student's understanding of the subject matter.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-exp-1206", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chain):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])
# Custom CSS for improved styling
st.markdown("""
<style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #ccc;
    }
    .stProgress>div>div>div {
        background-color: #4CAF50;
    }
    .stRadio > label {
        color: white !important;
    }
    .stRadio > div[role="radiogroup"] > label {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        display: block;
    }
    .stRadio > div[role="radiogroup"] > label:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def typewriter_effect(text):
    html_code = f"""
    <style>
        @keyframes type {{
            from {{ width: 0; }}
            to {{ width: 100%; }}
        }}
        @keyframes blink-caret {{
            from, to {{ border-color: transparent; }}
            50% {{ border-color: orange; }}
        }}
        .typewriter-text {{
            overflow: hidden;
            border-right: .15em transparent;
            white-space: nowrap;
            letter-spacing: .15em;
            animation:
                type 3.5s steps(40, end),
                blink-caret 0.75s step-end infinite;
            font-size: 2.5em;
            color: #ffffff;
            text-align: center; 
            display: block; 
            width: fit-content; 
            margin-left: auto;
            margin-right: auto; 
        }}
    </style>
    <h1 class="typewriter-text">{text}</h1>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def typewriter_effect_small(text):
    time.sleep(2.0)
    html_code = f"""
    <style>
        @keyframes type {{
            from {{ width: 0; }}
            to {{ width: 100%; }}
        }}
        @keyframes blink-caret {{
            from, to {{ border-color: transparent; }}
            50% {{ border-color: orange; }}
        }}
        .typewriter-text-small {{
            overflow: hidden;
            border-right: .15em transparent;
            white-space: nowrap;
            margin: 0 auto;
            letter-spacing: .15em;
            animation:
                type 3.5s steps(40, end),
                blink-caret 0.75s step-end 7; /* Blink for around 5 seconds */
            font-size: 1.2em; /* Smaller font size */
            color: #ffffff;
            text-align: center;
            display: block;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }}
    </style>
    <h1 class="typewriter-text-small">{text}</h1>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def process_screenshot_with_ocr_space(image):
    api_key = 'K82866588188957'  # Your OCR.space API key
    api_url = 'https://api.ocr.space/parse/image'

    try:
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Prepare files and headers
        files = {
            'file': ('image.jpg', image_bytes, 'image/jpeg')
        }
        payload = {
            'apikey': api_key,
            'language': 'eng',
            'OCREngine': '2'
        }
        
        # Make the request
        response = requests.post(api_url, files=files, data=payload)
        
        # Check if response is successful
        response.raise_for_status()
        
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError:
            st.error("Failed to decode API response")
            return "Error: Could not process the image. Please try again."

        # Check if the OCR was successful
        if result.get('OCRExitCode') != 1:
            error_message = result.get('ErrorMessage', ['Unknown error'])
            if isinstance(error_message, list):
                error_message = " ".join(error_message)
            st.error(f"OCR Error: {error_message}")
            return f"Error processing image: {error_message}"

        # Extract text from results
        parsed_text = ""
        if 'ParsedResults' in result and len(result['ParsedResults']) > 0:
            parsed_text = result['ParsedResults'][0].get('ParsedText', '').strip()
            
            if not parsed_text:
                return "No text was extracted from the image. Please try a clearer image."

            # Process with Gemini
            model = ChatGoogleGenerativeAI(model="gemini-exp-1206", temperature=0.7)
            response = model.invoke(parsed_text)
            return response.content if hasattr(response, 'content') else str(response)
        
        return "No text could be extracted from the image. Please try again with a clearer image."

    except requests.exceptions.RequestException as e:
        st.error(f"Network error occurred: {str(e)}")
        return "Error: Could not connect to the OCR service. Please check your internet connection and try again."
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return "An unexpected error occurred while processing the image. Please try again."

# PDF Processing Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Quiz Generation Functions
def generate_quiz(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = f"""
    Based on the following text, generate a quiz with 10 multiple-choice questions.
    For each question, provide one correct answer (prefixed with $) and three incorrect answers (prefixed with ~).
    Format each question EXACTLY as follows:

    Question: [Question text]
    $[Correct answer]
    ~[Incorrect answer 1]
    ~[Incorrect answer 2]
    ~[Incorrect answer 3]

    Ensure that each question follows this exact format, starting with 'Question:' and followed by four answer options.

    Text: {text[:1000]}  # Limit text to first 1000 characters to avoid token limit issues
    """
    try:
        response = model.invoke(prompt)
        quiz_text = response.content if hasattr(response, 'content') else str(response)
        print("Generated quiz (first 500 chars):", quiz_text[:500])
        return quiz_text
    except Exception as e:
        print(f"Error in generate_quiz: {str(e)}")
        return None

def parse_quiz(quiz_text):
    if not quiz_text:
        print("Quiz text is empty or None")
        return []
    
    questions = []
    current_question = None
    for line in quiz_text.split('\n'):
        line = line.strip()
        if line.startswith('Question:'):
            if current_question and current_question['answers']:
                questions.append(current_question)
            current_question = {'question': line[9:].strip(), 'answers': []}
        elif (line.startswith('$') or line.startswith('~')) and current_question is not None:
            current_question['answers'].append({
                'text': line[1:].strip(),
                'correct': line.startswith('$')
            })
    
    if current_question and current_question['answers']:
        questions.append(current_question)
    
    return questions

import random

def randomize_answers(questions):
    for question in questions:
        random.shuffle(question['answers'])
    return questions

def display_quiz_with_immediate_feedback(questions):
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = randomize_answers(questions)
        st.session_state.user_answers = {}
        st.session_state.submitted_answers = set()
        st.session_state.score = 0

    total_questions = len(st.session_state.quiz_questions)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, q in enumerate(st.session_state.quiz_questions, 1):
        progress = (i - 1) / total_questions
        progress_bar.progress(progress)
        progress_text.text(f"Question {i} of {total_questions}")
        
        st.subheader(f"Question {i}")
        st.write(q['question'])
        
        answer_key = f"answer_{i}"
        st.session_state.user_answers[answer_key] = st.radio(
            "Select your answer:", 
            [a['text'] for a in q['answers']], 
            key=f"q{i}",
            index=None  # Ensure no option is pre-selected
        )

        if st.button("Submit Answer", key=f"submit_answer{i}"):
            st.session_state.submitted_answers.add(i)

        if i in st.session_state.submitted_answers:
            correct_answer = next(a for a in q['answers'] if a['correct'])
            user_answer = st.session_state.user_answers[answer_key]
            
            st.write(f"Your answer: {user_answer}")
            st.write(f"Correct answer: {correct_answer['text']}")
            
            if user_answer == correct_answer['text']:
                st.success("Correct!")
                if i not in st.session_state.score_tracked:
                    st.session_state.score += 1
                    st.session_state.score_tracked.add(i)
                st.write("Explanation: Great job! You selected the correct answer.")
            else:
                st.error("Incorrect")
                st.write("Explanation: The answer you selected was not correct. Here's why the correct answer is right:")
            
            st.write(f"Rationale: The correct answer is the most appropriate based on the information in the text.")
        
        st.markdown("---")
        
        progress = i / total_questions
        progress_bar.progress(progress)
        progress_text.text(f"Question {i} of {total_questions}")

    progress_bar.progress(1.0)
    progress_text.text(f"Quiz Completed: {total_questions} of {total_questions}")
    
    st.subheader("Quiz Complete!")
    st.write(f"Your final score: {st.session_state.score} out of {total_questions}")


def main():
    st.header("AI Powered Study Helper 💁")
    typewriter_effect("StudyBuddi")
    typewriter_effect_small("Your perfect study partner")
    st.sidebar.image("https://assets.softr-files.com/applications/d89f3026-eb15-4c34-84f7-ef7818fa2c08/assets/a33eeaf3-3997-418b-b3bc-60b12d329b18.png",width=200,)

    with st.sidebar:
        st.title("")
        mode = st.radio("Choose mode:", ("PDF Analyzer", "Quiz Generator", "Screenshot Analyzer"))
        if st.button("Video Summarizer"):
            st.write("[Open Video Summarizer](http://localhost:8502)")
        if st.button("Repo Chat"):
            st.write("[Open Repo Chat](https://chat-with-repo.onrender.com/)")
        st.title("Upload PDF")
        pdf_docs = st.file_uploader("Drop your PDF files here", accept_multiple_files=True)
        process_button = st.button("Process PDF")

    if process_button and pdf_docs:
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.session_state.raw_text = raw_text
            st.success("PDF processed successfully!")

    if mode == "PDF Analyzer":
        st.subheader("PDF Analyzer")
        user_question = st.text_input("Ask a question about the PDF content:")
        if user_question:
            chain = get_conversational_chain()
            user_input(user_question, chain)

    elif mode == "Screenshot Analyzer":
        st.subheader("Screenshot Analyzer")
        screenshot = st.file_uploader("Upload a screenshot of a question", type=["png", "jpg", "jpeg"])
        if screenshot:
            image = Image.open(screenshot).convert('RGB')
            with st.spinner("Analyzing screenshot..."):
                answer = process_screenshot_with_ocr_space(image)
                st.write("Answer:", answer)

    else:  # Quiz Generator mode
        st.subheader("Quiz Generator")
        if st.button("Generate Quiz"):
            if 'raw_text' in st.session_state:
                with st.spinner("Generating quiz..."):
                    quiz_text = generate_quiz(st.session_state.raw_text)
                    if quiz_text:
                        questions = parse_quiz(quiz_text)
                        if questions:
                            st.session_state.quiz_questions = randomize_answers(questions)
                            st.session_state.user_answers = {}
                            st.session_state.submitted_answers = set()
                            st.session_state.score = 0
                            st.session_state.score_tracked = set()
                            st.success("Quiz generated successfully!")
                        else:
                            st.error("Failed to parse quiz questions. Please try again.")
                    else:
                        st.error("Failed to generate quiz. Please try again.")
            else:
                st.error("Please process a PDF first before generating a quiz.")
        
        if 'quiz_questions' in st.session_state:
            display_quiz_with_immediate_feedback(st.session_state.quiz_questions)

    st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Made with ❤️ ")

if __name__ == "__main__":
    main()