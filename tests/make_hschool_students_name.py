"""
generate_highschool_students_600.py

Generates 600 high school student records: 300 male, 300 female,
with random first and last names, grades, and unique IDs.
"""

import random

# Sample first names
male_first_names = [
    "James","John","Robert","Michael","William","David","Richard","Joseph",
    "Thomas","Charles","Christopher","Daniel","Matthew","Anthony","Mark","Donald",
    "Steven","Paul","Andrew","Joshua","Kenneth","Kevin","Brian","George","Edward",
    "Ronald","Timothy","Jason","Jeffrey","Ryan","Jacob","Gary","Nicholas","Eric",
    "Jonathan","Stephen","Larry","Justin","Scott","Brandon","Benjamin","Samuel",
    "Gregory","Alexander","Frank","Patrick","Raymond","Jack","Dennis","Jerry",
    "Tyler","Aaron","Jose","Adam","Henry","Nathan","Douglas","Zachary","Peter",
    "Kyle","Walter","Ethan","Jeremy","Harold","Keith","Christian","Roger","Noah",
    "Gerald","Carl","Terry","Sean","Austin","Arthur","Lawrence","Jesse","Dylan",
    "Bryan","Joe","Jordan","Billy","Bruce","Albert","Willie","Gabriel","Logan",
    "Alan","Juan","Wayne","Roy","Ralph","Randy","Eugene","Vincent","Russell","Elijah"
]

female_first_names = [
    "Mary","Patricia","Jennifer","Linda","Elizabeth","Barbara","Susan","Jessica",
    "Sarah","Karen","Nancy","Margaret","Lisa","Betty","Dorothy","Sandra","Ashley",
    "Kimberly","Donna","Emily","Carol","Michelle","Amanda","Melissa","Deborah",
    "Stephanie","Rebecca","Laura","Sharon","Cynthia","Kathleen","Amy","Shirley",
    "Angela","Helen","Anna","Brenda","Pamela","Nicole","Emma","Samantha","Katherine",
    "Christine","Debra","Rachel","Catherine","Carolyn","Janet","Ruth","Maria",
    "Heather","Diane","Virginia","Julie","Joyce","Victoria","Olivia","Kelly","Christina",
    "Lauren","Joan","Evelyn","Judith","Megan","Cheryl","Andrea","Hannah","Jacqueline",
    "Martha","Madison","Anna","Frances","Gloria","Ann","Teresa","Kathryn","Sara","Janice",
    "Jean","Alice","Abigail","Julia","Judy","Sophia","Grace","Denise","Amber","Doris",
    "Marilyn","Danielle","Beverly","Isabella","Theresa","Diana","Natalie","Brittany",
    "Charlotte","Marie","Kayla","Alexis","Lori","Jacquelyn","Holly","Audrey"
]

# Sample last names
last_names = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson","White",
    "Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker","Young",
    "Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores","Green",
    "Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter",
    "Roberts","Gomez","Phillips","Evans","Turner","Diaz","Parker","Cruz",
    "Edwards","Collins","Reyes","Stewart","Morris","Morales","Murphy","Cook",
    "Rogers","Gutierrez","Ortiz","Morgan","Cooper","Peterson","Bailey","Reed",
    "Kelly","Howard","Ramos","Kim","Cox","Ward","Richardson","Watson","Brooks",
    "Chavez","Wood","James","Bennet","Gray","Mendoza","Ruiz","Hughes","Price",
    "Alvarez","Castillo","Sanders","Patel","Myers","Long","Ross","Foster","Jimenez"
]

def generate_students(num_male=300, num_female=300):
    students = []
    
    # Generate male students
    for i in range(1, num_male+1):
        first = random.choice(male_first_names)
        last = random.choice(last_names)
        grade = random.randint(9,12)
        student_id = f"M{i:03d}"
        students.append({
            "id": student_id,
            "first_name": first,
            "last_name": last,
            "gender": "Male",
            "grade": grade
        })
    
    # Generate female students
    for i in range(1, num_female+1):
        first = random.choice(female_first_names)
        last = random.choice(last_names)
        grade = random.randint(9,12)
        student_id = f"F{i:03d}"
        students.append({
            "id": student_id,
            "first_name": first,
            "last_name": last,
            "gender": "Female",
            "grade": grade
        })
    
    # Shuffle to mix male and female
    random.shuffle(students)
    return students

if __name__ == "__main__":
    student_list = generate_students(300,300)
    for s in student_list:
        print(f"{s['id']} - {s['first_name']} {s['last_name']} ({s['gender']}, Grade {s['grade']})")