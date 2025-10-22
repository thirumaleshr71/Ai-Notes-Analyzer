from flask import Flask, render_template, request, jsonify
import os
from project1 import load_notes, abstractive_summarize, answer_question, generate_quiz

app = Flask(__name__, template_folder="ui_files", static_folder="static")

notes_text = ""

@app.route("/")
def index():
    return render_template("project1ui.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global notes_text
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename
    temp_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)

    try:
        notes_text = load_notes(temp_path)
        summary = abstractive_summarize(notes_text)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)

@app.route("/ask", methods=["POST"])
def ask():
    global notes_text
    question = request.json.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = answer_question(question, notes_text)
    return jsonify({"answer": answer})

@app.route("/quiz", methods=["GET"])
def quiz():
    global notes_text
    quizzes = generate_quiz(notes_text, num_questions=5)
    return jsonify({"quiz": quizzes})

if __name__ == "__main__":
    app.run(debug=True)
