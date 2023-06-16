# gebeta's version for grid print of the dataset:

import os
from tabulate import tabulate

# Define the directory path
directory = "/data/yarcoh/thesis_data/data/datasets"

# Initialize a dictionary to store the counts
counts = {}

# Read the directory
for folder in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, folder)):
        # Split the folder name into parts
        parts = folder.split('_')
        if len(parts) == 2:
            samples, mus = parts[1].split('x')
            key = (samples, mus)
            counts[key] = counts.get(key, 0) + 1

# Check if there are any counts
if not counts:
    print("No matching folders found.")
else:
    # Find the unique samples and mus values
    samples_list = sorted(set(key[0] for key in counts), key=int)
    mus_list = sorted(set(key[1] for key in counts), key=int)

    # Create the matrix
    matrix = []
    for sample in samples_list:
        row = [sample]
        for mus in mus_list:
            count = counts.get((sample, mus), 0)
            cell_value = "" if count == 0 else count
            row.append(cell_value)
        matrix.append(row)
    
    # Prepare the table headers
    headers = ['Samples \\ mu'] + mus_list

    # Generate the table using tabulate
    table = tabulate(matrix, headers, tablefmt="pretty")

    # Print the table
    print(table)
