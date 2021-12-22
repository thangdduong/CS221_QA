from typing import ContextManager
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

model_checkpoint = "nguyenvulebinh/vi-mrc-base"
nlp = pipeline('question-answering', model=model_checkpoint,
                tokenizer=model_checkpoint)

def qa_answer(question, context):
    QA_input = {
        "question": question,
        "context": context 
    }
    answer = nlp(QA_input)

    return answer

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route('/question_answering_test', methods=["GET", "POST"])
def question_answering_test():
    if request.method == "POST":
        user_question = request.form["user-question"]
        user_context = request.form["context"]

        answer = qa_answer(user_question, user_context)
        print(answer)
        return render_template("index.html", user_question=user_question, user_context=user_context, answer_found=answer["answer"])
    else:
        return ("nothing")
    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)