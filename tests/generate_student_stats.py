import random
import csv
import math
from collections import Counter

male_first_pool = ["Pippit","Hawk","Tiago","Ozias","Chozyn"]
female_first_pool = ["Fia","Yuna","Areli","Arden","Soraya"]
middle_name_pool = ["James","Alexander","Benjamin","Elijah","Samuel","Grace","Rose","Claire","Lily","Mae","Eli","Nora","Faith","Reed","June"]
last_name_pool = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Martinez","Hernandez","Lopez","Gonzalez","Wilson"]
homeroom_list = [f"{i}{x}" for i in range(101,106) for x in ["A","B"]]
core_courses = ["Math","English","Science","History","PhysicalEd"]
electives = ["Art","Music","ComputerScience","Spanish","French","Economics","Drama","Robotics","Philosophy","Astronomy","CreativeWriting"]
clubs = ["Chess","Drama","Robotics","Soccer","Debate","Art","Music","Science","Math","Environmental"]
awards = ["HonorRoll","PerfectAttendance","MathOlympiad","ScienceFair","DebateChampion","ArtShowWinner"]
used_names = set()

def unique_name(sex):
    while True:
        first = random.choice(male_first_pool if sex=="Male" else female_first_pool)
        middle = random.choice(middle_name_pool)
        last = random.choice(last_name_pool)
        full_name = f"{first} {middle} {last}"
        if full_name not in used_names:
            used_names.add(full_name)
            return first, middle, last

def personality_type():
    return random.choice(["Analytical","Creative","Athletic","Social","Techie","Curious"])

def generate_student(code, sex):
    first, middle, last = unique_name(sex)
    prev_gpa = round(random.uniform(2.5,4.0),2)
    years = {}
    club_history = []
    award_history = []
    for yr in ["Freshman","Sophomore","Junior","Senior"]:
        courses = {}
        elective_choices = random.sample(electives,3)
        all_courses = core_courses + elective_choices
        for c in all_courses:
            difficulty = random.uniform(0.8,1.2)
            grade = round(min(4.0,max(1.5,prev_gpa*random.uniform(0.8,1.2)/difficulty)),2)
            if random.random()<0.05:
                grade = round(max(1.0,grade-1.0),2)
            courses[c]=grade
        year_gpa = round(sum(courses.values())/len(courses),2)
        prev_gpa = year_gpa
        years[yr]={"courses":courses,"year_gpa":year_gpa}
        new_club = random.choice(clubs)
        if new_club not in club_history:
            club_history.append(new_club)
        if random.random()<0.3:
            award = random.choice(awards)
            if award not in award_history:
                award_history.append(award)
    cumulative_gpa = round(sum(years[y]["year_gpa"] for y in years)/4,2)
    leadership = {c: random.choice(["President","VP","Captain","Secretary"]) for c in club_history if random.random()<0.2}
    scholarship = cumulative_gpa>=3.75 and random.randint(1000,1600)>=1400
    return {
        "id": code,
        "first_name": first,
        "middle_name": middle,
        "last_name": last,
        "sex": sex,
        "homeroom": random.choice(homeroom_list),
        "persona": personality_type(),
        "cumulative_gpa": cumulative_gpa,
        "SAT": random.randint(1000,1600),
        "ACT": random.randint(20,36),
        "attendance_pct": round(random.uniform(85,100),2),
        "clubs": "|".join(club_history),
        "awards": "|".join(award_history),
        "leadership": "|".join([f"{c}:{r}" for c,r in leadership.items()]),
        "scholarship": scholarship
    }

def student_generator(num_male=10000, num_female=10000):
    for i in range(1,num_male+1):
        yield generate_student(f"M{i:05d}","Male")
    for i in range(1,num_female+1):
        yield generate_student(f"F{i:05d}","Female")

def export_csv(filename="students_20000.csv"):
    fields=["id","first_name","middle_name","last_name","sex","homeroom","persona",
            "cumulative_gpa","SAT","ACT","attendance_pct","clubs","awards","leadership","scholarship"]
    with open(filename,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for student in student_generator():
            writer.writerow(student)

def mean(data): return sum(data)/len(data) if data else 0
def median(data):
    sorted_data = sorted(data)
    n=len(sorted_data)
    mid = n//2
    return (sorted_data[mid-1]+sorted_data[mid])/2 if n%2==0 else sorted_data[mid]
def mode(data):
    if not data: return None
    count = Counter(data)
    max_count = max(count.values())
    return [k for k,v in count.items() if v==max_count]
def variance(data):
    m = mean(data)
    return sum((x-m)**2 for x in data)/len(data) if data else 0
def std_dev(data): return math.sqrt(variance(data))

def filter_high_gpa(csv_file, threshold=3.5):
    high=[]
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["cumulative_gpa"])>=threshold:
                high.append(row)
    return high

if __name__=="__main__":
    export_csv()
    high_achievers = filter_high_gpa("students_20000.csv")
    gpas = [float(s["cumulative_gpa"]) for s in high_achievers]
    print(len(high_achievers), round(mean(gpas),2))