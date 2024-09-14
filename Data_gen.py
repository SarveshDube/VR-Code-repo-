import numpy as np
import pandas as pd
from smt.sampling_methods import LHS
from pathlib import PurePath
import os
import sys

#print("testi") # Using this just to test the code\

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


# generate samples using smt sampling _method

def samples_LHS(input_file, num_samples=1000):
    print('in lhs function')

    # Get the first row of the input file
    first_row = df.iloc[0]
    
    # Create xlimits array for LHS sampling
    xlimits = np.array([[value * 0.85, value * 1.15] for value in first_row])
    
    # Set up LHS sampling
    sampling = LHS(xlimits=xlimits, random_state=20240408)
    
    # Generate samples
    samples = sampling(num_samples)
    
    # Convert samples to DataFrame
    samples_df = pd.DataFrame(samples, columns=df.columns)
    
    return samples_df

def validate_samples(original_value, samples):
    lower_limit = original_value * 0.85
    upper_limit = original_value * 1.15
    
    within_range = np.logical_and(samples >= lower_limit, samples <= upper_limit)
    percentage_within_range = np.mean(within_range) * 100
    
    return percentage_within_range

if __name__ == "__main__":
#in this function I am trying to use the smt library to generate samples for single row and then validating them based on the range provided(+_15%)
# Set the number of samples to be generated
    num = 1000
    name = "LHS_sample"
    data_dir_in = "data"
# Generate samples using LHS
    df = pd.read_csv(input_file)
    samples_df = samples_LHS(input_file, num)

# Validate results
    original_df = pd.read_csv(input_file)
    first_row = original_df.iloc[0]

    validation_results = {}
    for column in samples_df.columns:
        percentage = validate_samples(first_row[column], samples_df[column])
        validation_results[column] = percentage

    # Print validation results
    print("Validation Results:")
    for column, percentage in validation_results.items():
        print(f"{column}: {percentage:.2f}% within ±15% range")