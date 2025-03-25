from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import json
import fitz  # PyMuPDF for PDF extraction
import faiss
import numpy as np
import requests
import logging
from werkzeug.utils import secure_filename
from config import GROQ_API_KEY, SECRET_KEY

from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = SECRET_KEY
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
QUESTION_FOLDER = os.path.join(DATA_FOLDER, 'questions')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(QUESTION_FOLDER, exist_ok=True)

# Load users from JSON
def load_users():
    users_file = os.path.join(DATA_FOLDER, 'users.json')
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            return json.load(f)
    return {}

# Save users to JSON
def save_users(users):
    with open(os.path.join(DATA_FOLDER, 'users.json'), 'w') as f:
        json.dump(users, f, indent=4)

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    users = load_users()
    if data['username'] in users and users[data['username']]['password'] == data['password']:
        session['user'] = data['username']
        session.modified = True  # Ensure session updates
        logging.debug(f"Session after login: {session}")  # Log session data
        if data['username'] == 'admin':
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        else:
            return jsonify({'success': True, 'redirect': url_for('user_dashboard')})
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    # Basic validation
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'})
    
    # Load existing users
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return jsonify({'success': False, 'message': 'Username already exists'})
    
    # Create new user
    users[username] = {
        'password': password,
        'quizzes': {}
    }
    
    # Save updated users
    save_users(users)
    
    # Log in the new user
    session['user'] = username
    session.modified = True
    
    return jsonify({'success': True, 'redirect': url_for('user_dashboard')})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    logging.debug(f"Extracted text from {pdf_path}: {text[:500]}")  # Log first 500 chars for debugging
    return text

# Store embeddings using FAISS
def create_faiss_index(texts):
    dimension = 768  # Assume embedding size (adjust as needed)
    index = faiss.IndexFlatL2(dimension)
    vectors = np.random.rand(len(texts), dimension).astype('float32')  # Replace with real embeddings
    index.add(vectors)
    faiss.write_index(index, os.path.join(DATA_FOLDER, 'vector_index.faiss'))
def generate_mcq(text):
    logging.debug(f"Sending text to LLM: {text[:500]}")  # Log first 500 chars for debugging
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    # Enhanced prompt with clear JSON structure instructions
    prompt = (
        "Generate a comprehensive quiz with EXACTLY 10 multiple-choice questions in STRICT JSON format. "
        "JSON STRUCTURE MUST BE:\n"
        "{\n"
        "  \"questions\": [\n"
        "    {\n"
        "      \"question\": \"Question text\",\n"
        "      \"answers\": {\n"
        "        \"a\": \"Option A text\",\n"
        "        \"b\": \"Option B text\",\n"
        "        \"c\": \"Option C text\",\n"
        "        \"d\": \"Option D text\"\n"
        "      },\n"
        "      \"correct\": \"Correct option letter (a/b/c/d)\",\n"
        "      \"difficulty\": \"Low/Medium/High\"\n"
        "    },\n"
        "    ... (9 more questions)\n"
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        "- Cover topics from the context\n"
        "- 4 Low difficulty questions\n"
        "- 3 Medium difficulty questions\n"
        "- 3 High difficulty questions\n"
        "- Include numerical and conceptual questions\n"
        "- Ensure no duplicate questions\n"
        "- Numerical questions must show calculation method\n\n"
        f"Context:\n{text}\n\n"
        "IMPORTANT: Return ONLY valid JSON. No additional text before or after JSON."
    )
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an expert economics MCQ generator. Generate precise, educational multiple-choice questions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"}  # Explicit JSON request
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        response_data = response.json()
        
        # Extract MCQ text
        mcq_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Clean and prepare JSON
        mcq_text = mcq_text.strip()
        mcq_text = mcq_text.replace("```json", "").replace("```", "").strip()
        
        # Robust JSON parsing with error handling
        try:
            mcqs = json.loads(mcq_text)
        except json.JSONDecodeError as e:
            logging.error(f"JSON Parsing Error: {e}")
            logging.error(f"Problematic JSON: {mcq_text}")
            return {"error": f"Failed to parse JSON: {str(e)}"}
        
        # Validate MCQ structure
        if not isinstance(mcqs, dict) or "questions" not in mcqs:
            return {"error": "Invalid MCQ JSON structure"}
        
        # Ensure exactly 10 questions
        questions = mcqs.get("questions", [])
        if len(questions) != 10:
            return {"error": f"Expected 10 questions, got {len(questions)}"}
        
        return mcqs
    
    except requests.RequestException as e:
        logging.error(f"API Request Error: {e}")
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session or session['user'] != 'admin':
        return redirect(url_for('index'))
    
    quizzes = [f for f in os.listdir(QUESTION_FOLDER) if f.endswith('.json')]
    return render_template('dashboard.html', quizzes=quizzes)

@app.route('/user_dashboard')
def user_dashboard():
    if 'user' not in session:
        logging.warning("User not in session, redirecting to index")
        return redirect(url_for('index'))
    
    users = load_users()
    if session['user'] not in users:
        logging.warning(f"User {session['user']} not found in user database")
        return redirect(url_for('index'))

    quizzes = [f for f in os.listdir(QUESTION_FOLDER) if f.endswith('.json')]
    return render_template('user_dashboard.html', quizzes=quizzes)


@app.route('/all_quizzes')
def all_quizzes():
    if 'user' not in session or session['user'] != 'admin':
        return redirect(url_for('index'))
    
    quizzes = [f for f in os.listdir(QUESTION_FOLDER) if f.endswith('.json')]
    return render_template('all_quizzes.html', quizzes=quizzes)

@app.route('/delete_quiz/<filename>', methods=['POST'])
def delete_quiz(filename):
    if 'user' not in session or session['user'] != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized access'}), 403
    
    try:
        # Secure the filename to prevent path traversal attacks
        filename = secure_filename(filename)
        filepath = os.path.join(QUESTION_FOLDER, filename)
        
        # Check if the file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Quiz not found'}), 404
        
        # Delete the quiz file
        os.remove(filepath)
        
        # You might want to also update the users.json to remove references to this quiz
        users = load_users()
        for username, user_data in users.items():
            if 'quizzes' in user_data and filename in user_data['quizzes']:
                del user_data['quizzes'][filename]
        
        # Save the updated users data
        save_users(users)
        
        return jsonify({'success': True})
    
    except Exception as e:
        logging.error(f"Error deleting quiz: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user' not in session or session['user'] != 'admin':
        return jsonify({'error': 'Unauthorized access'})
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    text = extract_text_from_pdf(filepath)
    if not text.strip():
        logging.error("Extracted text is empty. Check PDF content.")
        return jsonify({'error': 'Extracted text is empty'})
    create_faiss_index([text])
    mcqs = generate_mcq(text)
    question_filepath = os.path.join(QUESTION_FOLDER, f"{os.path.splitext(filename)[0]}.json")

    with open(question_filepath, 'w') as f:
        json.dump(mcqs, f, indent=4)
    return jsonify({'success': True, 'mcqs': mcqs})

@app.route('/quiz/<filename>')
def quiz(filename):
    filepath = os.path.join(QUESTION_FOLDER, filename)
    if not os.path.exists(filepath):
        return "Quiz not found", 404
    with open(filepath, 'r') as f:
        quiz_data = json.load(f)
    return render_template('quiz.html', quiz=quiz_data, quiz_file=filename)




@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.json
    quiz_file = data.get("quiz_file")
    user_answers = data.get("answers")

    if 'user' not in session:
        return jsonify({"error": "User not logged in"}), 403

    username = session['user']
    
    filepath = os.path.join(QUESTION_FOLDER, quiz_file)
    if not os.path.exists(filepath):
        return jsonify({"error": "Quiz not found"}), 404

    with open(filepath, 'r') as f:
        quiz_data = json.load(f)

    # Extract correct answers
    correct_answers = {q["question"]: q["correct"] for q in quiz_data.get("questions", [])}

    score = 0
    incorrect_answers = []

    for idx, (question, user_option_key) in enumerate(user_answers.items()):
        correct_option_key = correct_answers.get(question)  

        user_answer_text = quiz_data["questions"][idx]["answers"].get(user_option_key, "Unknown")
        correct_answer_text = quiz_data["questions"][idx]["answers"].get(correct_option_key, "Unknown")

        if user_option_key == correct_option_key:
            score += 1  
        else:
            incorrect_answers.append({
                "question": question,
                "correct_answer": correct_answer_text,
                "user_answer": user_answer_text
            })

    # Load users.json and update it
    users_filepath = os.path.join(DATA_FOLDER, 'users.json')
    with open(users_filepath, 'r') as f:
        users = json.load(f)

    if username not in users:
        return jsonify({"error": "User not found"}), 404

    # Initialize quizzes dictionary if not present
    if "quizzes" not in users[username]:
        users[username]["quizzes"] = {}

    # Store scores under the quiz filename key
    if quiz_file not in users[username]["quizzes"]:
        users[username]["quizzes"][quiz_file] = []

    # Append the new quiz attempt
    users[username]["quizzes"][quiz_file].append({
        "score": score,
        "total": len(correct_answers),
    })

    # Save back to users.json
    with open(users_filepath, 'w') as f:
        json.dump(users, f, indent=4)

    return jsonify({"score": score, "total": len(correct_answers), "incorrect_answers": incorrect_answers})


@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('index'))

    users = load_users()
    username = session['user']

    if username not in users:
        return redirect(url_for('index'))

    # Get user's quiz data directly
    user_quizzes = users[username].get("quizzes", {})
    
    return render_template('profile.html', username=username, progress=user_quizzes)

@app.route('/users')
def users():
    if 'user' not in session or session['user'] != 'admin':
        return redirect(url_for('index'))
    
    # Load users from JSON file
    users = load_users()
    
    return render_template('users.html', users=users)

@app.route('/edit_quiz/<filename>')
def edit_quiz(filename):
    if 'user' not in session or session['user'] != 'admin':
        return redirect(url_for('index'))
    
    filepath = os.path.join(QUESTION_FOLDER, filename)
    if not os.path.exists(filepath):
        return "Quiz not found", 404
    
    return render_template('edit_quiz.html')

@app.route('/get_quiz/<filename>')
def get_quiz(filename):
    if 'user' not in session or session['user'] != 'admin':
        return jsonify({'error': 'Unauthorized access'}), 403
    
    filepath = os.path.join(QUESTION_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Quiz not found'}), 404
    
    with open(filepath, 'r') as f:
        quiz_data = json.load(f)
    
    return jsonify(quiz_data)

@app.route('/save_quiz/<filename>', methods=['POST'])
def save_quiz(filename):
    if 'user' not in session or session['user'] != 'admin':
        return jsonify({'error': 'Unauthorized access'}), 403
    
    filepath = os.path.join(QUESTION_FOLDER, filename)
    
    try:
        data = request.json
        
        # Validate data structure
        if not data or not isinstance(data, dict) or 'questions' not in data:
            return jsonify({'success': False, 'error': 'Invalid data format'}), 400
        
        # Validate each question has the required fields
        for i, question in enumerate(data['questions']):
            required_fields = ['question', 'answers', 'correct', 'difficulty']
            for field in required_fields:
                if field not in question:
                    return jsonify({
                        'success': False, 
                        'error': f'Question {i+1} is missing the required field: {field}'
                    }), 400
            
            # Validate answers contains options a, b, c, d
            for option in ['a', 'b', 'c', 'd']:
                if option not in question['answers']:
                    return jsonify({
                        'success': False, 
                        'error': f'Question {i+1} is missing option {option}'
                    }), 400
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        return jsonify({'success': True})
    
    except Exception as e:
        logging.error(f"Error saving quiz: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)