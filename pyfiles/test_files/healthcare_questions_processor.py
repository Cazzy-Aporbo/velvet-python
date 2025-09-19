import os
import pandas as pd
import re
import csv
import json
from collections import defaultdict

class HealthcareQuestionsProcessor:
    """
    A class to process and categorize healthcare questions from various sources
    and generate a comprehensive CSV file with appropriate metadata.
    """
    
    def __init__(self, output_dir="healthcare_data"):
        """Initialize the processor with output directory for data"""
        self.output_dir = output_dir
        self.sources_dir = os.path.join(output_dir, "sources")
        self.data_dir = os.path.join(output_dir, "data")
        
        # Create necessary directories
        os.makedirs(self.sources_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define colors for different healthcare professional types
        self.colors = {
            "Primary Care Physician": "#4285F4",  # Blue
            "Pediatrician": "#0F9D58",  # Green
            "OB-GYN": "#DB4437",  # Red
            "Women's Health Specialist": "#F4B400",  # Yellow
            "Cardiologist": "#FF6D01",  # Orange
            "Anesthesiologist": "#46BDC6",  # Teal
            "Dentist": "#9C27B0",  # Purple
            "Nurse": "#795548",  # Brown
            "Nurse Practitioner": "#607D8B",  # Blue Grey
            "Physician Assistant": "#009688",  # Teal
            "Specialist": "#673AB7",  # Deep Purple
            "Various": "#3F51B5",  # Indigo
            "Mental Health Provider": "#E91E63",  # Pink
            "Psychiatrist": "#E91E63",  # Pink
            "Psychotherapist": "#C2185B",  # Dark Pink
            "Psychologist": "#880E4F",  # Very Dark Pink
            "Therapist": "#AD1457",  # Medium Dark Pink
            "Dermatologist": "#8BC34A",  # Light Green
            "Ophthalmologist": "#FFC107",  # Amber
            "Pulmonologist": "#03A9F4",  # Light Blue
            "Respiratory Therapist": "#00BCD4",  # Cyan
            "ENT": "#FF9800",  # Deep Orange
            "Neurologist": "#9E9E9E",  # Grey
            "Orthopedist": "#FFEB3B",  # Yellow
            "Urologist": "#CDDC39",  # Lime
            "Gastroenterologist": "#FF5722",  # Deep Orange
        }
        
        # Initialize data structures
        self.questions_data = []
    
    def add_question_from_text(self, text, healthcare_type, visit_type, source, audience="General Adult Population"):
        """
        Extract questions from text and add them to the dataset
        """
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        questions = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and ('?' in sentence or any(q_word in sentence.lower() for q_word in ['what', 'how', 'when', 'where', 'why', 'can', 'should', 'do', 'is', 'are'])):
                # Clean up the question
                q = re.sub(r'\s+', ' ', sentence).strip()
                
                # Ensure it ends with a question mark if it looks like a question
                if not q.endswith('?') and any(q_word in q.lower() for q_word in ['what', 'how', 'when', 'where', 'why', 'can', 'should', 'do', 'is', 'are']):
                    q += '?'
                
                # Add if meets minimum length
                if len(q) > 10:
                    questions.append(q)
        
        # Add questions to our data structure
        for question in questions:
            self.add_question(question, healthcare_type, visit_type, source, audience)
        
        return questions
    
    def add_question(self, question, healthcare_type, visit_type, source, audience="General Adult Population"):
        """Add a single question to the dataset"""
        # Determine color based on healthcare professional type
        primary_type = healthcare_type.split('/')[0].strip()
        color = self.colors.get(primary_type, "#3F51B5")  # Default to Indigo if type not found
        
        self.questions_data.append({
            'Question': question,
            'Healthcare Professional Type': healthcare_type,
            'Visit Type': visit_type,
            'Color': color,
            'Source': source,
            'Target Audience': audience
        })
    
    def add_questions_from_markdown(self, markdown_file):
        """
        Parse a markdown file with questions and add them to the dataset
        Expected format:
        # Title
        
        ## Source
        URL
        
        ## Target Audience
        Audience
        
        ## Healthcare Professional Type
        Type
        
        ## Questions by Visit Type
        
        ### Visit Type 1
        1. Question 1
        2. Question 2
        
        ### Visit Type 2
        - Question 3
        - Question 4
        """
        try:
            with open(markdown_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract source
            source_match = re.search(r'## Source\s*\n(.*?)(?:\n|$)', content)
            source = source_match.group(1).strip() if source_match else "Unknown"
            
            # Extract target audience
            audience_match = re.search(r'## Target Audience\s*\n(.*?)(?:\n|$)', content)
            audience = audience_match.group(1).strip() if audience_match else "General"
            
            # Extract healthcare professional type
            hcp_match = re.search(r'## Healthcare Professional Type\s*\n(.*?)(?:\n|$)', content)
            hcp_type = hcp_match.group(1).strip() if hcp_match else "Various"
            
            # Extract visit types and questions
            visit_sections = re.findall(r'### (.*?)\n((?:- .*?\n|[0-9]+\. .*?\n)+)', content)
            
            for visit_type, questions_text in visit_sections:
                # Extract individual questions
                question_lines = re.findall(r'(?:- |[0-9]+\. )(.*?)(?:\n|$)', questions_text)
                for question in question_lines:
                    question = question.strip()
                    if question:
                        self.add_question(question, hcp_type, visit_type, source, audience)
            
            print(f"Added questions from {markdown_file}")
            
        except Exception as e:
            print(f"Error processing markdown file {markdown_file}: {e}")
    
    def process_directory(self, directory=None):
        """Process all markdown files in a directory"""
        if directory is None:
            directory = self.sources_dir
        
        for filename in os.listdir(directory):
            if filename.endswith('.md'):
                file_path = os.path.join(directory, filename)
                self.add_questions_from_markdown(file_path)
    
    def save_to_csv(self, filename="healthcare_questions.csv"):
        """Save all collected questions to CSV"""
        output_path = os.path.join(self.data_dir, filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Question', 'Healthcare Professional Type', 'Visit Type', 'Color', 'Source', 'Target Audience']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for question_data in self.questions_data:
                writer.writerow(question_data)
        
        print(f"CSV file created successfully with {len(self.questions_data)} questions at {output_path}")
        return output_path
    
    def load_from_csv(self, filepath):
        """Load questions from an existing CSV file"""
        try:
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                self.questions_data.append({
                    'Question': row['Question'],
                    'Healthcare Professional Type': row['Healthcare Professional Type'],
                    'Visit Type': row['Visit Type'],
                    'Color': row['Color'],
                    'Source': row['Source'],
                    'Target Audience': row['Target Audience']
                })
            print(f"Loaded {len(df)} questions from {filepath}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
    
    def get_statistics(self):
        """Get statistics about the collected questions"""
        if not self.questions_data:
            return "No questions collected yet."
        
        df = pd.DataFrame(self.questions_data)
        
        stats = {
            "Total questions": len(df),
            "Unique healthcare professional types": df["Healthcare Professional Type"].unique().tolist(),
            "Unique visit types": df["Visit Type"].unique().tolist(),
            "Questions per healthcare type": df["Healthcare Professional Type"].value_counts().to_dict(),
            "Questions per visit type": df["Visit Type"].value_counts().to_dict()
        }
        
        return stats
    
    def export_statistics(self, filename="statistics.json"):
        """Export statistics to a JSON file"""
        stats = self.get_statistics()
        output_path = os.path.join(self.data_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics exported to {output_path}")
        return output_path
    
    def categorize_questions(self):
        """
        Categorize questions by topic using keyword matching
        Adds a 'Category' field to each question
        """
        # Define categories and their keywords
        categories = {
            "Diagnosis": ["diagnosis", "symptoms", "condition", "problem", "disease", "disorder"],
            "Treatment": ["treatment", "medication", "medicine", "drug", "therapy", "surgery", "procedure"],
            "Prevention": ["prevent", "avoid", "risk", "protect", "healthy", "lifestyle"],
            "Follow-up": ["follow up", "follow-up", "next appointment", "check-up", "checkup"],
            "Side Effects": ["side effect", "reaction", "complication"],
            "Lifestyle": ["diet", "exercise", "activity", "food", "nutrition", "sleep"],
            "Insurance/Cost": ["insurance", "cost", "price", "afford", "coverage", "pay"],
            "Credentials": ["credential", "experience", "training", "education", "qualified"],
            "Scheduling": ["schedule", "appointment", "visit", "time", "available"],
            "Emergency": ["emergency", "urgent", "worsen", "worse", "severe"]
        }
        
        # Categorize each question
        for question_data in self.questions_data:
            question_text = question_data['Question'].lower()
            
            # Count matches for each category
            category_matches = defaultdict(int)
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword.lower() in question_text:
                        category_matches[category] += 1
            
            # Assign the category with the most keyword matches
            if category_matches:
                best_category = max(category_matches.items(), key=lambda x: x[1])[0]
                question_data['Category'] = best_category
            else:
                question_data['Category'] = "General"
        
        # Update CSV fieldnames for future saves
        print(f"Categorized {len(self.questions_data)} questions")
    
    def filter_questions(self, healthcare_type=None, visit_type=None, category=None):
        """Filter questions by healthcare type, visit type, or category"""
        filtered_data = self.questions_data.copy()
        
        if healthcare_type:
            filtered_data = [q for q in filtered_data if healthcare_type.lower() in q['Healthcare Professional Type'].lower()]
        
        if visit_type:
            filtered_data = [q for q in filtered_data if visit_type.lower() in q['Visit Type'].lower()]
        
        if category and 'Category' in self.questions_data[0]:
            filtered_data = [q for q in filtered_data if q.get('Category', '').lower() == category.lower()]
        
        return filtered_data
    
    def save_filtered_csv(self, filename, healthcare_type=None, visit_type=None, category=None):
        """Save filtered questions to a CSV file"""
        filtered_data = self.filter_questions(healthcare_type, visit_type, category)
        
        output_path = os.path.join(self.data_dir, filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(filtered_data[0].keys()) if filtered_data else ['Question', 'Healthcare Professional Type', 'Visit Type', 'Color', 'Source', 'Target Audience']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for question_data in filtered_data:
                writer.writerow(question_data)
        
        print(f"Filtered CSV file created with {len(filtered_data)} questions at {output_path}")
        return output_path


# Example usage
def main():
    # Initialize the processor
    processor = HealthcareQuestionsProcessor()
    
    # Create example markdown files
    create_example_markdown_files(processor.sources_dir)
    
    # Process all markdown files in the sources directory
    processor.process_directory()
    
    # Add some manual questions
    processor.add_question(
        "How often should I get a mammogram?",
        "OB-GYN",
        "Women's Health Checkup",
        "Manual Entry",
        "Adult Women"
    )
    
    processor.add_question(
        "What vaccines does my child need before starting school?",
        "Pediatrician",
        "Well Child Visit",
        "Manual Entry",
        "Parents"
    )
    
    # Categorize questions
    processor.categorize_questions()
    
    # Save to CSV
    csv_path = processor.save_to_csv()
    
    # Export statistics
    stats_path = processor.export_statistics()
    
    # Create filtered CSVs for specific healthcare types
    processor.save_filtered_csv("primary_care_questions.csv", healthcare_type="Primary Care")
    processor.save_filtered_csv("mental_health_questions.csv", healthcare_type="Mental Health")
    processor.save_filtered_csv("specialist_questions.csv", healthcare_type="Specialist")
    
    # Print statistics
    print("\nStatistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value) if len(value) < 10 else f'{len(value)} unique values'}")
        elif isinstance(value, dict):
            print(f"{key}: {len(value)} categories")
        else:
            print(f"{key}: {value}")
    
    print(f"\nCSV file saved to: {csv_path}")
    print(f"Statistics saved to: {stats_path}")


def create_example_markdown_files(sources_dir):
    """Create example markdown files for demonstration"""
    
    # Primary Care example
    primary_care_md = """# Questions to Ask Your Primary Care Provider

## Source
https://example.com/primary-care-questions

## Target Audience
General Adult Population

## Healthcare Professional Type
Primary Care Physician

## Questions by Visit Type

### Annual Checkup
1. How are my vital signs compared to last year?
2. Are there any screening tests I should have based on my age and risk factors?
3. Do I need any vaccinations?
4. How is my weight and BMI?
5. Should I make any changes to my diet or exercise routine?

### Medication Review
- Are there any medications I should stop taking?
- Could any of my medications be causing side effects?
- Are there generic alternatives to my current medications?
- How do my medications interact with each other?
"""
    
    # Mental Health example
    mental_health_md = """# Questions to Ask Your Mental Health Provider

## Source
https://example.com/mental-health-questions

## Target Audience
General Adult Population

## Healthcare Professional Type
Mental Health Provider

## Questions by Visit Type

### First Appointment
1. What is your approach to therapy?
2. How often will we meet?
3. How long will therapy take?
4. How will we measure progress?
5. What is your experience with my specific concerns?

### Medication Discussion
- What are the potential side effects of this medication?
- How long will it take to see results?
- Will I need to take this medication long-term?
- How does this medication work?
"""
    
    # Write the example files
    with open(os.path.join(sources_dir, "primary_care_example.md"), "w", encoding="utf-8") as f:
        f.write(primary_care_md)
    
    with open(os.path.join(sources_dir, "mental_health_example.md"), "w", encoding="utf-8") as f:
        f.write(mental_health_md)
    
    print("Created example markdown files in sources directory")


if __name__ == "__main__":
    main()
