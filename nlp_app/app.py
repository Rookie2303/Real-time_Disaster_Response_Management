from flask import Flask, render_template, request, redirect, url_for, jsonify, Blueprint
from textblob import TextBlob

# app = Flask(__name__)

nlp_app = Blueprint('nlp_app', __name__, template_folder='templates', static_folder='static')

def analyze_text(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.3:
        return "Severe"
    elif analysis.sentiment.polarity < 0:
        return "Low"
    else:
        return "Moderate"

@nlp_app.route('/')
def index():
    return render_template('index_nlp.html')

@nlp_app.route('/get_prediction', methods=['GET', 'POST'])
def get_prediction():
    if request.method == 'POST':
        description = request.form.get('description')
        if description:
            return redirect(url_for('nlp_app.predict', description=description))
        else:
            return render_template('index_nlp.html', error="Please enter a description.")
    return redirect(url_for('nlp_app.index'))


# @nlp_app.route('/get_prediction', methods=['POST'])
# def get_prediction():
#     description = request.form['description']
    
#     if description:
#         return redirect(url_for('nlp_app.predict', description=description))
#     else:
#         return render_template('index_nlp.html', error="Please enter a description.")

@nlp_app.route('/predict')
def predict():
    description = request.args.get('description')
    
    if description:
        severity = analyze_text(description)
        return render_template('predict.html', severity=severity, description=description)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(nlp_app)
    app.run(debug=True)

