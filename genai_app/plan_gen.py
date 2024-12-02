import cohere

co = cohere.Client('gw2xsCZBJBc55nhH1KKuALzgKlgn5Ps5uiiaulAp')

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
        max_tokens=300
    )
    return response.generations[0].text

# Usage
disaster_type = "Wildfire"
severity = "High"
location = "California"
plan = generate_disaster_plan(disaster_type, severity, location)
print("Disaster Plan:\n", plan)
