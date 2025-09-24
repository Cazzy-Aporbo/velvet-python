import random
import requests

API_KEY = "YOUR_API_KEY"
API_URL = "https://api.api-ninjas.com/v1/babynames"

def fetch_baby_names(gender, limit=100):
    headers = {"X-Api-Key": API_KEY}
    params = {"gender": gender, "limit": limit}
    try:
        response = requests.get(API_URL, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return [item['name'] for item in data]
    except:
        return []

middle_names = ["James", "Alexander", "Benjamin", "Elijah", "Samuel",
                "Grace", "Rose", "Claire", "Lily", "Mae"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones",
              "Garcia", "Martinez", "Hernandez", "Lopez", "Gonzalez"]
predicted_male_names = ["Pippit", "Hawk", "Tiago", "Ozias", "Chozyn"]
predicted_female_names = ["Fia", "Yuna", "Areli", "Arden", "Soraya"]

male_names = fetch_baby_names("male", 100)
female_names = fetch_baby_names("female", 100)
male_names.extend(predicted_male_names)
female_names.extend(predicted_female_names)

def generate_students(num_male=300, num_female=300):
    students = []
    for i in range(1, num_male + 1):
        first = random.choice(male_names)
        middle = random.choice(middle_names)
        last = random.choice(last_names)
        student_id = f"M{i:03d}"
        students.append({"id": student_id,"first_name": first,"middle_name": middle,"last_name": last,"gender": "Male"})
    for i in range(1, num_female + 1):
        first = random.choice(female_names)
        middle = random.choice(middle_names)
        last = random.choice(last_names)
        student_id = f"F{i:03d}"
        students.append({"id": student_id,"first_name": first,"middle_name": middle,"last_name": last,"gender": "Female"})
    random.shuffle(students)
    return students

if __name__ == "__main__":
    student_list = generate_students(300, 300)
    for s in student_list:
        print(f"{s['id']} - {s['first_name']} {s['middle_name']} {s['last_name']} ({s['gender']})")