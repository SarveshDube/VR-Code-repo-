import numpy as np
import pandas as pd
from smt.sampling_methods import LHS
from pathlib import PurePath
import os
import sys

print("testi") # Using this just to test the code\

#Reading the input file 
input_file= pd.read_csv('/Users/sarveshdube/Documents/VR documents/code/filter_CO.csv')
# display file content
print(input_file.head)


def generate_samples_with_adjustments(input_file, num1=10, num2=10):
    
    
    # Load the input CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)
    
    # Create an empty list to store the samples
    samples = []
    
    # Loop over each row in the DataFrame
    for index, row in df.iterrows():
        # For each value in the row, apply a ±15% range
        adjusted_samples = []
        for value in row:
            lower_limit = value * 0.85  # 15% lower
            upper_limit = value * 1.15  # 15% upper
            # Generate a random sample within the range
            sample = np.random.uniform(lower_limit, upper_limit)
            adjusted_samples.append(sample)
        
        # Append the generated sample row to the samples list
        samples.append(adjusted_samples)
    
    # Convert the list of samples back into a DataFrame
    sample_df = pd.DataFrame(samples, columns=df.columns)
    
    return sample_df

# Example usage
input_file = "/Users/sarveshdube/Documents/VR documents/code/filter_CO.csv"
samples_df = generate_samples_with_adjustments(input_file)

# Save the generated samples to a new CSV file if needed
output_file= "output_samples.csv"
samples_df.to_csv(output_file, index=False)

# Function to print the newly generated samples
def print_generated_samples(output_file):
    if os.path.exists(output_file):
        samples_df = pd.read_csv(output_file)
        print("Generated Samples:")
        print(samples_df)
    else:
        print(f"The file {output_file} does not exist.")

# Print the samples to verify
print_generated_samples(output_file)