import os
import pandas as pd
import random
import numpy as np

def swap_random_values(df, column_index, swap_fraction=0.05):
    """
    Randomly swaps values in a column for a given fraction of rows.
    """
    num_rows = len(df)
    num_swaps = int(num_rows * swap_fraction)
    if num_swaps < 1:
        return df  # Skip if there are too few rows to swap

    # Select random pairs of indices to swap
    indices = np.random.choice(num_rows, num_swaps * 2, replace=False)
    for i in range(0, len(indices), 2):
        idx1, idx2 = indices[i], indices[i + 1]
        df.iloc[idx1, column_index], df.iloc[idx2, column_index] = (
            df.iloc[idx2, column_index],
            df.iloc[idx1, column_index],
        )
    return df

def process_parquet_files(input_dir, output_dir, swap_fraction=0.1):
    """
    Processes all parquet files in the input directory, swaps values in each column,
    and saves the modified dataframes to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".parquet"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Read the parquet file
            df_original = pd.read_parquet(input_path)

            # Make a copy of the original dataframe for comparison
            df_swapped = df_original.copy()

            # Precompute column indices for faster access
            column_indices = {column: df_original.columns.get_loc(column) for column in df_original.columns}

            # Swap values in each column
            for column, column_index in column_indices.items():
                print(f"    Processing column: {column}")
                df_swapped = swap_random_values(df_swapped, column_index, swap_fraction)

            # Compare the original and swapped dataframes for 10 random sets of 5 columns
            num_columns = len(df_original.columns)
            if num_columns < 5:
                print(f"File: {file_name} - Not enough columns to select sets of 5.")
                continue

            # Precompute the differences once for all columns
            differences = (df_original != df_swapped)

            differing_fractions = []
            for _ in range(10):
                selected_columns = random.sample(list(df_original.columns), 5)
                num_differing_rows = differences[selected_columns].any(axis=1).sum()
                differing_fraction = num_differing_rows / len(df_original)
                differing_fractions.append(differing_fraction)

            # Calculate the average fraction of differing rows
            average_differing_fraction = sum(differing_fractions) / len(differing_fractions)
            print(f"File: {file_name} - Average fraction of differing rows (10 sets of 5 columns): {average_differing_fraction:.2%}")

            # Save the modified dataframe to the output directory
            df_swapped.to_parquet(output_path, index=False)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_directory = "original_data_parquet"
    output_directory = "weak_data_parquet"
    process_parquet_files(input_directory, output_directory)