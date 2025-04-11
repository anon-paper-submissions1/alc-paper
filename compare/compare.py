import sys
import os
from alc_attacks.best_row_match.brm_attack import BrmAttack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

orig_files_dir = os.path.join('..', 'original_data_parquet')
anon_files_dir = os.path.join('..', 'anon_data_parquet')
work_files_dir = os.path.join('work_files')
os.makedirs(work_files_dir, exist_ok=True)
os.makedirs('slurm_out', exist_ok=True)


def do_work(job_num):
    print(f"Job number {job_num} started.")
    files_list = os.listdir(orig_files_dir)
    files_list.sort()
    print(files_list)
    files_list_index = job_num % len(files_list)
    file_name = files_list[files_list_index]
    print(f"Processing file {file_name} with index {files_list_index}")
    my_file = os.path.join(orig_files_dir, file_name)
    # read in my_file parquet file as pandas dataframe 
    df_orig = pd.read_parquet(my_file)
    # read in the corresponding anonymized file 
    my_file_anon = os.path.join(anon_files_dir, file_name)
    df_anon = pd.read_parquet(my_file_anon)
    df_cntl = df_orig.sample(n=1000, random_state=42)
    df_orig = df_orig.drop(df_cntl.index)
    print(f"Created df_cntl with {len(df_cntl)} rows")
    print(f"Original df has {len(df_orig)} rows")
    # strip suffix .parquet from file_name
    file_name = file_name.split('.')[0]
    attack_dir_name = f"{file_name}.{job_num}"
    my_work_files_dir = os.path.join(work_files_dir, attack_dir_name)
    os.makedirs(my_work_files_dir, exist_ok=True)
    brm = BrmAttack(df_original=df_orig,
                        df_control=df_cntl,
                        df_synthetic=df_anon,
                        results_path=my_work_files_dir,
                        attack_name = attack_dir_name,
                        )
    brm.run_auto_attack()


def do_plot():
    pass

import os
import pandas as pd

def do_gather():
    # List to store dataframes
    dataframes = []
    
    # Recursively walk through the directory
    for root, _, files in os.walk(work_files_dir):
        for file in files:
            if file == "summary_secret_known.csv":
                # Construct the full file path
                file_path = os.path.join(root, file)
                
                # Read the file into a dataframe
                df = pd.read_csv(file_path)
                
                # Append the dataframe to the list
                dataframes.append(df)
    
    print(f"Found {len(dataframes)} files named 'summary_secret_known.csv'.")
    # Concatenate all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Write the combined dataframe to a parquet file
        combined_df.to_parquet("all_secret_known.parquet", index=False)
        print("Parquet file written: all_secret_known.parquet")
    else:
        print("No files named 'summary_secret_known.csv' were found.")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'measure':
            if len(sys.argv) != 3:
                print("Usage: dependence.py measure <job_num>")
                quit()
            do_work(int(sys.argv[2]))
        elif sys.argv[1] == 'plot':
            do_plot()
        elif sys.argv[1] == 'gather':
            do_gather()
    else:
        print("No command line parameters were provided.")

if __name__ == "__main__":
    main()