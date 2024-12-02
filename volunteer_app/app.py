from flask import Flask, render_template, request, redirect, url_for, Blueprint
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from .volunteer_management import Volunteer, Task, VolunteerManagementSystem

# app = Flask(__name__)

volunteer_app = Blueprint('volunteer_app', __name__, template_folder='templates', static_folder='static')

system = VolunteerManagementSystem()

@volunteer_app.route('/')
def home():
    return render_template('index_volunteer.html')

@volunteer_app.route('/add_volunteer', methods=['POST'])
def add_volunteer():
    name = request.form['name']
    skills = set(request.form['skills'].split(','))
    availability = request.form.get('availability') == 'on'
    city = request.form['city']
    volunteer = Volunteer(name, skills, availability, city)
    system.add_volunteer(volunteer)
    return redirect(url_for('volunteer_app.home'))

@volunteer_app.route('/add_task', methods=['POST'])
def add_task():
    task_name = request.form['task_name']
    required_skills = set(request.form['required_skills'].split(','))
    city = request.form['city']
    task = Task(task_name, required_skills, city)
    system.add_task(task)
    return redirect(url_for('volunteer_app.home'))

@volunteer_app.route('/assign')
def assign():
    assignments = system.assign_volunteers()
    assignment_messages = [
        f"Assigning {name} to {task} in {city}" if name else f"No available volunteer for {task} in {city}"
        for name, task, city in assignments
    ]
    return render_template('assignments.html', assignments=assignment_messages)

if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(volunteer_app)
    app.run(debug=True)