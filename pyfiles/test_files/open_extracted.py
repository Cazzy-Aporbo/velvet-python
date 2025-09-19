import re
import csv

# Open the extracted text file
with open('extracted_text.txt', 'r') as file:
    lines = file.readlines()

# Initialize an empty list to store the extracted data
extracted_data = []

# Define a regular expression pattern to match the date and time
pattern_date_time = r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}'

# Define a regular expression pattern to match the winning numbers
pattern_winning_numbers = r'\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}'

# Define a regular expression pattern to match the bullseye
pattern_bullseye = r'\d{1,2}\s+X\d'

# Iterate over the lines and extract the relevant data
for line in lines:
    # Check if the line contains date and time
    if re.match(pattern_date_time, line):
        date_time = line.strip()
    # Check if the line contains winning numbers
    elif re.match(pattern_winning_numbers, line):
        winning_numbers = ' '.join(line.strip().split())
    # Check if the line contains bullseye
    elif re.match(pattern_bullseye, line):
        bullseye = line.strip()
        # Once we have all three pieces of data, append to the list
        extracted_data.append((date_time, winning_numbers, bullseye))

# Write the extracted data to a CSV file
with open('updated_keno_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(['Date', 'Winning Numbers', 'BullsEye'])
    # Write the data
    for data in extracted_data:
        writer.writerow(data)

print('Data has been written to updated_keno_data.csv.')