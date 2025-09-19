import os
import pandas as pd

def create_periodic_table_csv(file_path):
    elements_data = [
        {'name': 'Hydrogen', 'symbol': 'H', 'atomic_number': 1},
        {'name': 'Helium', 'symbol': 'He', 'atomic_number': 2},
        {'name': 'Lithium', 'symbol': 'Li', 'atomic_number': 3},
        {'name': 'Beryllium', 'symbol': 'Be', 'atomic_number': 4},
        {'name': 'Boron', 'symbol': 'B', 'atomic_number': 5},
        {'name': 'Carbon', 'symbol': 'C', 'atomic_number': 6},
        {'name': 'Nitrogen', 'symbol': 'N', 'atomic_number': 7},
        {'name': 'Oxygen', 'symbol': 'O', 'atomic_number': 8}
    ]

    df = pd.DataFrame(elements_data)

    df = df.append([
        {'name': 'Fluorine', 'symbol': 'F', 'atomic_number': 9},
        {'name': 'Neon', 'symbol': 'Ne', 'atomic_number': 10}
    ])

    df['atomic_weight'] = df['atomic_number'].apply(round)

    # Write DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(df)

    df_read = pd.read_csv(file_path)

    print("\nData read from CSV file:")
    print(df_read)

# Provide file path and filename with the ".csv" extension
file_path = "/Users/cazandraaporbo/Desktop/Summer/data_viz/table_of_elements.csv"

# Create the directory if it doesn't exist
directory = os.path.dirname(file_path)
os.makedirs(directory, exist_ok=True)

# Call the function with the specified file path
create_periodic_table_csv(file_path)

