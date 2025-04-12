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
plots_dir = os.path.join('plots')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs('slurm_out', exist_ok=True)
os.makedirs('slurm_prior_out', exist_ok=True)


def do_work(job_num, measure_type):
    work_files_dir = os.path.join('work_files')
    if measure_type == 'prior_measure':
        work_files_dir = os.path.join('work_files_prior')
    os.makedirs(work_files_dir, exist_ok=True)
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
    df_cntl = df_orig.sample(n=1000)
    df_orig = df_orig.drop(df_cntl.index)
    print(f"Created df_cntl with {len(df_cntl)} rows")
    print(f"Original df has {len(df_orig)} rows")
    # strip suffix .parquet from file_name
    file_name = file_name.split('.')[0]
    attack_dir_name = f"{file_name}.{job_num}"
    if measure_type == 'prior_measure':
        df_orig = df_anon.copy()
    my_work_files_dir = os.path.join(work_files_dir, attack_dir_name)
    os.makedirs(my_work_files_dir, exist_ok=True)
    brm = BrmAttack(df_original=df_orig,
                        df_control=df_cntl,
                        df_synthetic=df_anon,
                        results_path=my_work_files_dir,
                        attack_name = attack_dir_name,
                        )
    brm.run_auto_attack()


def do_plots():
    # read in all_secret_known.parquet file as pandas dataframe
    df = pd.read_parquet("all_secret_known.parquet")
    # make a new column called alc_floor where all alc values less than 0 are set to 0
    df['alc_floor'] = df['alc'].clip(lower=-0.5)
    idx = df.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_best = df.loc[idx].reset_index(drop=True)
    df_one = df[df['attack_recall'] == 1].reset_index(drop=True)
    idx = df_one.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_one = df_one.loc[idx].reset_index(drop=True)
    # for each combination of secret_column and known_columns, that is in both
    # df_best and df_one, make a new dataframe df_diff that contains the difference
    # of 'alc' between df_best and df_one
    # Merge df_best and df_one on 'secret_column' and 'known_columns'
    df_merged = pd.merge(
        df_best[['secret_column', 'known_columns', 'alc_floor']],
        df_one[['secret_column', 'known_columns', 'alc_floor']],
        on=['secret_column', 'known_columns'],
        suffixes=('_best', '_one')
    )
    # Compute the difference of alc values
    df_merged['alc_difference'] = df_merged['alc_floor_best'] - df_merged['alc_floor_one']
    print("df_best alc:")
    print(df_best['alc_floor'].describe())
    print("df_one alc:")
    print(df_one['alc_floor'].describe())
    # describe alc_difference
    print(df_merged['alc_difference'].describe())

    # make df_top which is df_merged where alc_floor >= 0.5
    df_top = df_merged[df_merged['alc_floor_best'] >= 0.5]
    print(f"There are {len(df_top)} rows with alc >= 0.5")
    print(f"Top group described:")
    print(df_top['alc_difference'].describe())

    # make a seaborn scatterplot from df_top with alc_floor_best on x and alc_floor_one on y
    plt.figure(figsize=(6, 3))
    sns.scatterplot(data=df_top, y='alc_floor_best', x='alc_floor_one')
    plt.ylabel('Worst-case ALC with recall')
    plt.xlabel('ALC without recall')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.55, 1.05)
    # draw a diagonal line from (0,0) to (1,1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dotted', linewidth=0.5)
    # draw a horizontal line at y=0.5
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=0.8)
    plt.axhline(y=0.75, color='red', linestyle='--', linewidth=0.8)
    # draw a vertical line at x=0.5
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=0.8)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True)
    # tighten
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'alc_best_vs_one.png'))
    plt.savefig(os.path.join(plots_dir, 'alc_best_vs_one.pdf'))
    plt.close()

def do_gather(measure_type):
    print(f"Gathering files for {measure_type}...")
    work_files_dir = os.path.join('work_files')
    out_name = 'all_secret_known.parquet'
    if measure_type == 'prior_measure':
        work_files_dir = os.path.join('work_files_prior')
        out_name = 'all_secret_known_prior.parquet'
    # List to store dataframes
    dataframes = []
    
    # Recursively walk through the directory
    for root, _, files in os.walk(work_files_dir):
        for file in files:
            if file == "summary_secret_known.csv":
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                dir_name = os.path.dirname(file_path)
                dir_name = os.path.basename(dir_name)
                dir_name = dir_name.split('.')[0]
                print(f"Adding dir_name {dir_name} to dataframe")
                df['dataset'] = dir_name
                dataframes.append(df)
    
    print(f"Found {len(dataframes)} files named 'summary_secret_known.csv'.")
    # Concatenate all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Write the combined dataframe to a parquet file
        combined_df.to_parquet(out_name, index=False)
        print(f"Parquet file written: {out_name}")
    else:
        print("No files named 'summary_secret_known.csv' were found.")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'measure':
            if len(sys.argv) != 3:
                print("Usage: dependence.py measure <job_num>")
                quit()
            do_work(int(sys.argv[2]), 'measure')
        elif sys.argv[1] == 'prior_measure':
            if len(sys.argv) != 3:
                print("Usage: dependence.py measure <job_num>")
                quit()
            do_work(int(sys.argv[2]), 'prior_measure')
        elif sys.argv[1] == 'plot':
            do_plots()
        elif sys.argv[1] == 'gather':
            do_gather('measure')
            do_gather('prior_measure')
    else:
        print("No command line parameters were provided.")

if __name__ == "__main__":
    main()