from flask import Flask, request, render_template, redirect, url_for, Blueprint
import cohere

# Initialize the Cohere client
co = cohere.Client('gw2xsCZBJBc55nhH1KKuALzgKlgn5Ps5uiiaulAp')

genai_app = Blueprint('genai_app', __name__, template_folder='templates', static_folder='static')


# app = Flask(__name__)
def generate_disaster_plan(disaster_type, severity, location):
    prompt = f"""
    Disaster: {disaster_type}
    Severity: {severity}
    Location: {location}

    Provide a detailed plan of action for rescuing citizens effectively. Include:
    1. Immediate response steps.
    2. Resources needed.
    3. Communication strategies.
    4. Safety protocols.
    """    
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=200
    )
    return response.generations[0].text

@genai_app.route('/')
def index():
    return render_template('index_genai.html')

@genai_app.route('/generate_plan', methods=['POST'])
def generate_plan():
    # Retrieve form data
    disaster_type = request.form['disaster_type']
    severity = request.form['severity']
    location = request.form['location']

    # Generate the disaster plan
    plan = generate_disaster_plan(disaster_type, severity, location)

    # Redirect to the plan page with the generated plan
    return render_template('plan.html', plan=plan, disaster_type=disaster_type, severity=severity, location=location)

if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(genai_app)
    app.run(debug=True)
