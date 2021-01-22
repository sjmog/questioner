from flask import Flask, request, render_template
from question_generation_model.pipelines import pipeline

def generate_qa_pairs(summary):
    # convert the summary into question/answer pairs
    # summary => qa_pairs
    nlp = pipeline("question-generation", model="valhalla/t5-base-qa-qg-hl")

    qa_pairs = nlp(summary)

    def is_answer_too_long(qa_pair):
        return len(qa_pair["answer"]) > 30

    def filter_long_answers(qa_pairs):
        result = []
        for qa_pair in qa_pairs:
            if(is_answer_too_long(qa_pair)):
                continue
            result.append(qa_pair)
        return result

    return filter_long_answers(qa_pairs)

app = Flask(__name__)

@app.route('/')
def index():
    text = request.args.get('text')
    qa_pairs = None

    if(text):
        summary = text.replace("\n", "")
        qa_pairs = generate_qa_pairs(summary)

    return render_template('index.html', qa_pairs=qa_pairs)
