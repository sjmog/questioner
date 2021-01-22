from flask import Flask
from flask import request
from flask import render_template
from question_generation_model.pipelines import pipeline

app = Flask(__name__)

@app.route('/')
def index():
    text = request.args.get('text')
    qa_pairs = None

    if(text):
        summary = text.replace("\n", "")

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

        qa_pairs = filter_long_answers(qa_pairs)

        # for each question/answer pair
        # print the question/answer pair to the screen
        # qa_pair => (print side effect) => qa_pair

        # Check if the answer was right

    return render_template('index.html', qa_pairs=qa_pairs)
