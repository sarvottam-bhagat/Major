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
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
import re

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
    model = ChatGoogleGenerativeAI(model="gemini-exp-1206", temperature=0.7)
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

def extract_video_id(youtube_url):
    video_id_match = re.search(r'(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def generate_youtube_summary(transcript_text):
    if transcript_text is None:
        return "No transcript available to generate content."
    
    prompt = """You are Yotube video summarizer. You will be taking the transcript text
    and summarizing the entire video and providing the important summary in points
    within 500 words. Please provide the summary of the text given here: """
    
    model = ChatGoogleGenerativeAI(model="gemini-exp-1206", temperature=0.3)
    response = model.invoke(prompt + transcript_text)
    return response.content if hasattr(response, 'content') else str(response)

def main():
    st.header("AI Powered Study Helper 💁")
    typewriter_effect("StudyBuddi")
    typewriter_effect_small("Your perfect study partner")
    st.sidebar.image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBw8NEhAQEBAREBAVEBUVGRUVFRcQEhAWGBUWHRgWGRUYHykgGh0lHxcYITEhJiktLjoyFyA1ODMsNyktOisBCgoKDg0OGhAQFy0dHx0rLSsrLSstNy0tLS0tLS0tLS0tLS0tLSstLS0tNystLS0tLTctLSstLTctLS0tLS0tN//AABEIAMgAyAMBIgACEQEDEQH/xAAcAAEAAgMBAQEAAAAAAAAAAAAABgcDBQgEAQL/xABHEAABAwIBCAcEBwUGBwEAAAABAAIDBBEFBgcSITFRYYETIkFxcpGhMkJSsRQjYoKSssEzY6LC0VR0k9Lh8BYXNEODhLMV/8QAGQEAAgMBAAAAAAAAAAAAAAAAAwQAAgUB/8QAJREAAgIBBQACAgMBAAAAAAAAAAECAxEEEiExQTJRYXEUIkIT/9oADAMBAAIRAxEAPwC8URFCBERQgRFiqJ2RNc+RzWMaLlziGtaN5JUIZFhqqqOFpfI9kbBtc4hrRzKrjKbOkxl46Fgkds6V4IYPCzae827iq0xXF6mtdp1Er5Xdlz1W3+Fo1Dkjw08pcvgPDTyly+C4MYznUEF2xadS4fANFl/E79AVD8Szq10lxDHFAN9jK8c3avRQJEzGiC8GY0QXhu6zK/EpvbrJh4XdEPJllq5q6aT25ZHeJ7nfNYERFFLpBFFLpH3SO9Zoa6aP2JZG+F7m/JYEXTpu6PK/EofYrJj4ndKPJ91IsOzq10dhNHFON9uieeY1eigSKjri+0VdcX2i7cHznUE9my6dM4/GNJl/G39QFM6WpjmaHxvbIw7HNIc08wuX17cKxeponadPM+J3bY9V1viadR5oMtMv8gJaZP4nTK+Ks8mc6TH6MdcwRu2dKwEsPibtb3i47lY9NOyVrXxua9jhcOaQ5pG8EJWUHHsWlBx7MyIiqUCIihAiIoQIiKEPiIozlrldFhUfY+oeOpHf+N1tjR6/LsYtvCOxi28I9eVGU1NhkenK67yOpGPbkPd2DiqSynyqqsTfeV2jED1Ym6mN/qeJ3rWYniE1XI6aZ5fI46yfkB2DgvKn66VHn0frpUefQiIjBgiz0dHLUODIo3yPPusaXH07OKmuD5ra2azp3spm7j9bJ5DV6qkpxj2ykpxj2yBorqoM1dBHbpXTTntu7Qb5N1jzW7p8icLj2UcR8V5PzEoT1MV0Cepj4c9Iujf+FsO/sVN/hM/ovNUZEYXJtpIx4bx/kIVf5Mfor/Jj9HPaK6a/NXQSX6J00B7LO02+Ttfqoli+aythu6B8dS3d+yk8navVEjfB+hI3wfpA0Weso5adxjmjfE8e69pafXb3rAi5yFzkLeZMZVVWGOvE7SiJ60Tj1Hf5TxG5aNFGk1hkaTWGdFZMZTU+Jx6cLrPHtxnU+M93aOK3a5kwzEZqSRs0LyyRp1EdvAjtHBXrkVldFisfYyoYOvHf+Nu9p9PmjbTt5XQjbTt5XRJ0REAAERFCHxEWKpnZEx0j3BrGtLnE7AALkqENTlblFFhcDpX2c86mM7ZHfoBtJXP2J4hLVyvmmcXyPNyf0G4BbPLHKF+J1DpTcRjqxt+Fm+287T/otGtCmvauezQpr2rnsIiy0tM+Z7Y42l73GwaNZJRegpjaCbAC5J2DtVh5J5spZ9GWtLoY9oiH7V3i+Du29ylmQ2QkWHgTThstURt2th4M4/a8uM1Sll/kRSy/yJ4cKwmno2dHTxNibwGt3FztpPEr3IiV7Fc5PqIihAiIoQIiKEPBi2E09Yzo6iJkjeI1t4h20HiFReXmBQYbU9DDI57SwPIdrMdybNv27+YV+1EzYmPkebNa0uJ3AC5K5tx3EnVtRNUO2yPLrfCPdbyFhyTOmzn8DOmzn8HhRETo6F6sMxCWklZNC4tkabg7+BHaCvKi41ng41k6KySyiixOBszOq8ans7Y39o4jtBW7XO+RuUT8MqGyi5id1ZGj3m77bxtHPeug6edkrWyMIcxzQ4EbCCLgrPtr2P8ABn217H+DMiIhAj4qyzw5Q6DWUMZ1vs+S3Y2/VZzOvkN6sesqWwxvlebMYwuJ3AC5XNuNYk+snlqH+1I8utt0R2N5Cw5I+nhull+B9PDdLL8PEiInx8+taSQALknUBtKu/N3kc3D4xNMAap7df7lp9wcd55dmuKZpcmRPIa2Vt443WjB2Ok+L7vZx7lcCTvs/yhO+z/KPqIiVFT4sVRUMiaXyPaxgFy5xDWt7yVHcscs4MLbo/tagi7Ywdn2nH3R8/lS2P5RVWIv06iQkX6rB1Y2dzf12o1dLlz4GrpcufC2MYzn0FPdsQfUuHawaMd/G79AVFavO1VuP1VPBGPtaUp8wW/JV4iajRBeDUaILwm3/ADRxP9x/hn+t1cWEPmdDC6oDRMYwXhoLQ1xFyLHdsVGZvMH+m1sTSLxxnpX7rNtYcbu0RZdAJe9RTSSF71FNJI+oiJcXILnaxj6NSdA02fO7R4hgsX/yj7ypNSrOVjH0ytkDTeOH6pu7q30j+K45BRVaNMdsTRpjtiFscDwSeve+OBuk5kTpCN4FtQ4m4AWuVz5ocH6CldUuHXndq4RtuB5nSPku2z2RydtnsjkphwIJBFiOzcisPOzk0IJBWxNtHK60gGxsnxdzvmOKrxdhNSWTsJqSygrZzP5Q6bHUMh1sBfHftbfrN5E35ncqmXtwTEn0U8VQz2o3g22aQ95vMXHNcshujg5ZDdHB0wvqw0dQ2ZjJGG7HtDgd4IuCizTNIZncxToKMQtNnTvDeOg3rOP5R95Ukp5nir+krGQg6ooRq3Of1j6aCga0KI4gjQojiCCz0FI+okjhjF3yPDB3k2WBT3M9hfTVb53DqwR6vG+4HoHq85bYtl5y2xbLbwbDWUcEVPGOrGwN7z2uPEm55r2oizDM7CjGXeVTcLh6tnVD7iNu7e93AevykVVO2Jj5HkNYxpc4nYABclc6ZT40/EamSofcAmzG/AwX0W/77SUamvc+eg1Ne589Gvq6l8z3SSOL3uNy46ySsSInx8Ii9eEUDquaKBntSSBu+19p5DXyUbwiN4LczQ4P0FK6ocLPndq4RtuB5nSPkp8sNHTMhjZEwWYxjWgbg0AAeizrMnLdJszJy3SbPi0uWGL/AECknnv1w3RZxe7U31N+S3SqPPNjOnJDRtOqMdI/xOFmjk25++rVR3SSLVR3SSK2JvrOsoiLSNI9eEUDquaKBntSPDfDfaeQ18l0pR0zYI44mCzGMDANwaLBVRmZwfTllrHDVGOjZ43e0eTbD76t1I6iWZY+hHUSzLH0eHGsNZWwS08nsyMI8J7HDiDY8lzfXUj6eSSF4s9j3McOIJC6eVLZ4ML6GrZO0WbOy58bLA+mgu6aeHg7ppYeCBoiJ0dLtzR4r09F0Tjd0DyzjoHrN+bh91FD8zlf0dZJCTqlhOre5msemmizrltmzOuW2bI5lrVGavrH/v3N5M6o9GrSrNXS9JJI/wCKRzvMkrCn4rCwPxWFgK68z9F0dEZba5ZnG/2W9UDzDvNUouhsgoOjw+jG+EO/ES79UHUvEcAdS/64JAiIkREgud3FTBRiFps6d+jx0G9Z38o+8qTVhZ6aouqoIuxkGlze839GBV6tCiOIGhRHEEEREYMFZGZrB9OWWscOrGOjZ43DrHvDbfjVbgXXRWR2D/QKOCEiz9HSf43a3eWzkEDUSxHH2AvliOPs3iIiQEDBV1DYWPkebMY0uJ3AC5K5sxnEHVc81Q/2pJC622w7ByFhyVuZ3cY6ClFO02fO6x4MbYu8zojmVS6c00eMjmmjxkIiJoaLpzO1rZKJ0QADopnX3uDtYcfUfdU8VPZlaotqaiLsfAH82PA/nKuFZ1yxNmdcsTYUEzw0XSUTZba4pmm/2XXaR5lqnaj+X8HSYfWDdFpfgId+irW8STK1vEkc9IiLTNM3eRNV0NfRv/ftZyf1T+ZFq8Pl6OWJ/wAMjHeTgV9QLa9zyAtr3PJ5yiz18XRyys+GRzfIkLAjBgujskf+hov7rD/82rnFdC5AzdJh9Gd0Qb+Elv6JbU/FC2p+KJCiIkxMo7O9f/8AQ/8ABH83KFKw89NKW1NPL2Pg0ebHkn84VeLSp5gjSq5ggiIiBCUZt8H+m1sdxeOL6127q20R+K3kVftlSGbrK2mwvpWzRPvI4XlbZ1gBqaWbhcm4PbsVvYTjlLWt0qeZknAGzh3tOscwkdRucuuBHUbnLrg2KItFlrjP0CjmmBs/R0GeN2oeW3kgJZeACWXgp3OLjH02tlIN44/qm7rN2nm7S9FGUJRacY7VhGnGO1YQREVixN8z9/p//ryfNqu5U9mWpS6pqJexkAZze8H+Qq4Vn6j5iGo+YWoyv/6Gt/us35HLbqP5fz9Hh9Yd8Wj+Ihv6ocfkgUfkjnpERahqH1u0d6LNh8XSSxM+KRjfNwC+qrlgq5YNnltS9DX1jP37ncn9YfmWkU9zx0HR1jJgNUsI173M1H0LFAlWt5imcreYphXVmerekonRX1xTOFvsus4HzLvJUqp5mfxToat8DjZs7LDxsuR6F6rfHMCl8cwLqREWeZ5Bs7mFGoo+laLugeH8dA6nfyn7qpJdQ1ELZWujeA5rmlpB2EEWIXOmVOCPw6pkgdfRBuxx9+M30T+neCnNNPjaOaafG01KIiaGgv3DK6NwcxzmOB1FpLXDuIX4RcOE0wPOXX01myltVGPj6sluDxt5gr8Zf5ZNxUQMia+NjAXOa62t51dmogC+v7RUORU/5RTykU/5RTykEREQIERbbJXBH4jUxwNvo30nuHuMFtI/oOJC43hZZxvCyy2M0uFGnoulcLOneX8dAdVvyJ+8pusdPC2NrWMAaxrQ0AbAALALIsyUtzbMyUtzbCgeeKt6OibEDrlmaLfZb1ifMNU8VK538V6arbA03bAyx8b7Od6BivTHM0XpjmaIIiItE0Td5EUvTV9G3981/KPrH8qKRZnKDpKuSYjqxQnXuc82HoHokr5/2wJXz/sTDO5hfT0fTNF3QPDuOg7quH5T91Ukun6unbMx8bxdj2FpG8EWIXNuN4a+inlp3+1G8i+zSHuu5ix5q+mllbS+mlxtPEs1DVPp5I5WGz2PD2niCCsKJl8jL5Ol8FxJlbBFUR+zIwHwntb3g3HJe5U7mmym+jyGildaOV12E7GyH3e53zHFXEs2yGyWDOshslgKM5c5LMxSGws2dlzG7j2tPA+mpSZFVSaeUUjJp5RzBW0kkD3xSsLJGmzmnaD/AL+awroDLDI+nxRt3fVzgWbKBr8Lh7zeCpbKHJqrw52jPGQ2/Vkb1o39zuzuOtP12qX7H67VP9moRERgwREUIERbfJ7JqrxF2jBGdG+uR3VjZ3u7e4a1VtLlnG0uWa6ipJKh7IomF8jnWa0bSVfWQ+SzMLh0TZ077GR/HsaOA/UpkhkfT4W27frJ3CzpSLHwtHut4KSpK67dwuhG67dwuj6iIgADwY3ibKKCWok9mNhNviPut7ybDmubq2qfPJJK83e95eTvJJJ7lPM7OU30iQUcTrxxOu8jY6Td935k7lXqe09e1Zfo9p4bVl+hEXuwPDX1s8NOzbI8C+3RHvO5AEo7eOQ7eOS4M0mF/R6LpXCz53l/HQHVb8ifvIplS07YWMjYLMY0NA3ACwHkizJvc8mXN7nkzKss8GT2m1tdG3rMAZJbtZfqu5E25jcrMWOogZK10b2hzHNLSDsIIsQVIS2vJaEtrycvIt7llk6/DKh0RuYndaNx95u6+8bD/qtEtNNNZRpJprKAJGsairuzc5ZCvjEE7rVTG7T/AN5o94fa3jn3UislNUPhe2SNxY9pBa4GxBQ7K1NYB2VqaOoUUIyGy8jrw2CctjqrWHY2bi3c7e3y4TdISi4vDEJRcXhn1Yp4WSNLHta9pFi1wDgRxBWVFUqQjGM2dBUXdEH0zz8Buy/gd8hZRWrzSVTT9VUwvH2w6I+Q0lcCIsbpr0Krpr0pL/lXiW+n79N3+Ve2kzSVJ/bVMLB9gOlPkdFXAi7/ACJlv5EyE4Pmzw+nIdIH1Lh8Zsy/gbq5G6mUELI2hrGtY0CwDQGgDcANiyIhuTl2wTk5ds+oiKpU+KEZxsshQRmngdeqe3aP+w0+8ftbvPv+ZdZeR0AdBTlslVsOwsg4u3u+z58aWqah8r3SSOL3uJLnE3JPNM0055YzTTnl9GMm+3aiInR0K2sz+T3RsfXSDrPBZHfsZfrO5kAfdO9QPIzJ1+J1DYhcRNs6R3wt3X3nYP8ARdBU8DYmtjYA1jWhrQNgAFgErqLMLahXUWYW1GZERJiYREUIaTKzJ6LE4HQv6rxrY+2uN3YeI3hc/Yph0tJK+CZuhIw2I7DuI3grptRnLXJKLFY+xlQwdST+V29vy+Z6bdvD6D027eH0UAi9WKYdNSSOhmYWSNOw9vEHtHFeVPJ5Hk8gG2saj/vtVhZJ5zJqcCKsDp49gkH7Vo439scdveq9RVnBSXJWcFJcnS2E4xTVrNOnlbK3gdbeDmnWD3r3rmCkqpIHB8T3RvGxzXFpHMKaYRnQroLNmaypbvP1cn4m6vRKz0zXQrPTNdF1ooFh+dWgkt0rJoT3CRvm3X6Ld0+XOFybKuMeIOj/ADAILrkvALrkvCRItN/xXh39tpv8Vv8AVeaoy5wuPbVxnwh0n5QVzZL6ObJfRIkUCxDOrQx3ETJpj3CNvm7X6KIYxnQrp7thDKZu8fWSfidq8grxpm/C8aZvwt3FsZpqJmnUStjbxPWdwa0azyVVZWZzJqnSiow6CI6jIf2rxw+D59ygdVVSTuL5XukedrnEuJ5lYkzDTxjy+RmGnUeXyCb6ztRETAwF6sKw6WslZBC3TkebDcN5O4BMLw2askbDAwvkcdQHYN5PYOKvbIvJKLCo+x9Q4deS38Ldzfn8hW2qC/IK21QX5PXkpk9FhkDYWdZ51vf2yO7T3bgt0iLPbbeWZ7bbyz6iIuHAiIoQIiKENJlPk1T4nHoTNs8DqyD24z39o4KksqMk6rDHWlbpRE9WVvsO7/hPA7u1dDrFUQMkaWPa17HCxa4aTSNxBRa7XD9Ba7XD9HL6K2sps1scl5KF4jdt6J5JYfC7a3nfkqzxbBqmidoVEL4j2Ejqut8LhqPJOwtjLodhbGXR4UREQIERFCBERQgREUIERe7CcGqa12hTwvlPaQOq2/xO2DmuNpdnG8dnhW+yWyTqsTd9W3QiB60rh1G93xHgPRT3JjNbHHaSueJXbeiZcRjxO2u5W5qxaeFkbQxjWsYBYNaNEAbgAlrNQuoi1moS4iarJnJqnwyPQhbdx9qQ63yHid3BbpESjbbyxRtt5Z9REXDgREUIERFCBERQgREUIFhqaaOZpZIxsjDta4BzTyK+ooQhuL5ssPqLui06Z/2DpM/C75AhQ/Ec1NbHcwyRTjcbxPPI6vVERY3Tj6FjdOPpHKzI7Eofao5j4B0o/gutZNh08ftwys8THN+YRE1XY5djVdjl2YOid8J8lnhw6eT2IZX+FjnfIIiK3gK2bSjyOxKb2aOYeMCIfx2Uiw7NTWyWM0sUA3C8rxyFh6oiUnfLwUnfLwmGD5s8Pp7OlD6l4+M2Z+BvyN1MaanZE0MjY1jBsa0BrR3AakRLubk+RdzcuzMiIuHAiIoQIiKECIihD//Z",width=200,)

    with st.sidebar:
        st.title("")
        
        # Add CSS for buttons
        button_style = """
        <style>
        .sidebar-button {
            width: 100%;
            margin: 5px 0;
            padding: 10px;
            text-align: center;
            background-color: #4CAF50;
            color: white !important;
            border-radius: 8px;
            text-decoration: none !important;
            display: block;
            cursor: pointer;
            border: none;
            font-size: 1em;
        }
        .sidebar-button:hover {
            background-color: #45a049;
            text-decoration: none !important;
            color: white !important;
        }
        .sidebar-button.selected {
            background-color: #45a049;
            border: 2px solid white;
        }
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)
        
        st.markdown("### Choose Mode")
        
        # Initialize the mode in session state if not present
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = "Vector RAG"
        
        # Create mode selection buttons using Streamlit buttons
        if st.button("Vector RAG", key="vector_rag_btn", use_container_width=True):
            st.session_state.current_mode = "Vector RAG"
            
        if st.button("Quiz Generator", key="quiz_gen_btn", use_container_width=True):
            st.session_state.current_mode = "Quiz Generator"
            
        if st.button("Image Analyzer", key="img_analyzer_btn", use_container_width=True):
            st.session_state.current_mode = "Image Analyzer"
        
        if st.button("YouTube Analyzer", key="youtube_analyzer_btn", use_container_width=True):
            st.session_state.current_mode = "YouTube Analyzer"
        
        mode = st.session_state.current_mode
        
        # External links styled as buttons
        st.markdown("""
            <div style="margin-top: 20px;">
                <a href="https://chat-with-repo.onrender.com/" target="_blank" class="sidebar-button">Repo Chat</a>
            </div>
        """, unsafe_allow_html=True)
        
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

    if mode == "Vector RAG":
        st.subheader("Chat With Your Knowledge Base")
        user_question = st.text_input("Ask a question about the uploaded content:")
        if user_question:
            chain = get_conversational_chain()
            user_input(user_question, chain)

    elif mode == "Image Analyzer":
        st.subheader("Image Analyzer")
        screenshot = st.file_uploader("Upload a image of a question", type=["png", "jpg", "jpeg"])
        if screenshot:
            image = Image.open(screenshot).convert('RGB')
            with st.spinner("Analyzing screenshot..."):
                answer = process_screenshot_with_ocr_space(image)
                st.write("Answer:", answer)

    elif mode == "YouTube Analyzer":
        st.subheader("YouTube Video Analyzer")
        youtube_link = st.text_input("Enter YouTube Video Link:")

        if youtube_link:
            try:
                video_id = extract_video_id(youtube_link)
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                
                if st.button("Get Video Summary"):
                    with st.spinner("Analyzing video..."):
                        transcript_text = extract_transcript_details(youtube_link)
                        if transcript_text:
                            summary = generate_youtube_summary(transcript_text)
                            st.markdown("## Video Summary:")
                            st.write(summary)
                        else:
                            st.warning("No transcript available for this video.")
                        
            except ValueError:
                st.error("Invalid YouTube URL")

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


