from geopy.geocoders import Nominatim
from geopy.distance import geodesic

class Volunteer:
    def __init__(self, name, skills, availability, city):
        """
        name: Name of the volunteer.
        skills: A set of skills that the volunteer possesses.
        availability: Boolean indicating if the volunteer is available.
        city: Name of the city where the volunteer is located.
        """
        self.name = name
        self.skills = set(skills)
        self.availability = availability
        self.city = city
        self.location = self.get_lat_lon(city)

    def get_lat_lon(self, city):
        """Convert city name to latitude and longitude using geopy."""
        geolocator = Nominatim(user_agent="volunteer_app")
        location = geolocator.geocode(city)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Could not find location for {city}")
            return None

class Task:
    def __init__(self, task_name, required_skills, city):
        """
        task_name: The name or description of the task.
        required_skills: A set of skills required to complete the task.
        city: Name of the city where the task is located.
        """
        self.task_name = task_name
        self.required_skills = set(required_skills)
        self.city = city
        self.location = self.get_lat_lon(city)

    def get_lat_lon(self, city):
        """Convert city name to latitude and longitude using geopy."""
        geolocator = Nominatim(user_agent="task_app")
        location = geolocator.geocode(city)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Could not find location for {city}")
            return None

class VolunteerManagementSystem:
    def __init__(self):
        self.volunteers = []
        self.tasks = []

    def add_volunteer(self, volunteer):
        self.volunteers.append(volunteer)

    def add_task(self, task):
        self.tasks.append(task)

    def find_best_match(self, task):
        best_volunteer = None
        best_score = 0

        for volunteer in self.volunteers:
            if not volunteer.availability or volunteer.location is None or task.location is None:
                continue  # Skip if volunteer or task has no valid location

            # Calculate skill match score
            skill_match = len(volunteer.skills & task.required_skills) / len(task.required_skills)

            # Calculate proximity (distance in kilometers)
            distance = geodesic(volunteer.location, task.location).km
            proximity_score = max(0, 1 - (distance / 50))  # Normalizing proximity score

            # Total score = 70% skill match + 30% proximity
            total_score = (0.7 * skill_match) + (0.3 * proximity_score)

            if total_score > best_score:
                best_score = total_score
                best_volunteer = volunteer

        return best_volunteer

    def assign_volunteers(self):
        assignments = []
        for task in self.tasks:
            best_volunteer = self.find_best_match(task)

            if best_volunteer:
                assignments.append((best_volunteer.name, task.task_name, task.city))
                best_volunteer.availability = False
            else:
                assignments.append((None, task.task_name, task.city))
        return assignments
