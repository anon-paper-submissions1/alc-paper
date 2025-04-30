import sys
import os
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from anonymity_loss_coefficient import BrmAttack
from anonymity_loss_coefficient.utils import get_good_known_column_sets

pp = pprint.PrettyPrinter(indent=4)

orig_files_dir = os.path.join('..', 'original_data_parquet')
anon_files_dir = os.path.join('..', 'weak_data_parquet')
plots_dir = os.path.join('plots')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs('slurm_out', exist_ok=True)


def do_attack(job_num):
    # read in jobs.json
    with open('jobs.json', 'r') as f:
        jobs = json.load(f)
    # get the job from the jobs list
    if job_num >= len(jobs):
        print(f"Job number {job_num} is out of range. There are only {len(jobs)} jobs.")
        sys.exit()
    job = jobs[job_num]
    print(f"Job number {job_num} started.")
    pp.pprint(job)
    df_orig = pd.read_parquet(os.path.join(orig_files_dir, job['dataset']))
    if job['approach'] == 'ours':
        work_files_dir = os.path.join('work_files')
        use_anon_for_baseline = False
    else:
        work_files_dir = os.path.join('work_files_prior')
        use_anon_for_baseline = True
    os.makedirs(work_files_dir, exist_ok=True)
    # read in the corresponding anonymized file 
    df_anon = pd.read_parquet(os.path.join(anon_files_dir, job['dataset']))
    # strip suffix .parquet from file_name
    file_name = job['dataset'].split('.')[0]
    attack_dir_name = f"{file_name}.{job_num}"
    my_work_files_dir = os.path.join(work_files_dir, attack_dir_name)
    os.makedirs(my_work_files_dir, exist_ok=True)

    brm = BrmAttack(df_original=df_orig,
                    df_synthetic=df_anon,
                    results_path=my_work_files_dir,
                    attack_name = attack_dir_name,
                    use_anon_for_baseline=use_anon_for_baseline,
                    no_counter=True,
                    )
    if False:
        # for debugging
        brm.run_one_attack(
            secret_column='n_unique_tokens',
            known_columns=['n_tokens_title', 'n_tokens_content',],
        )
        quit()
    # get all columns in df_orig.columns but not in job['known_columns']
    secret_columns = [c for c in df_orig.columns if c not in job['known_columns']]
    # shuffle the secret columns, but seeded to insure that prior and ours use the same columns
    random.seed(42)
    random.shuffle(secret_columns)
    print(f"Secret columns: {secret_columns}")
    print(f"Known columns: {job['known_columns']}")
    # Select 5 random secret columns for the attacks
    for secret_column in secret_columns[:5]:
        print(f"Running attack for {secret_column}...")
        brm.run_one_attack(
            secret_column=secret_column,
            known_columns=job['known_columns'],
        )

def do_plots():
    plot_prior_versus_ours()
    quit()
    plot_alc_best_vs_one()

def plot_prior_versus_ours():
    # Read the parquet files into dataframes
    try:
        df_ours = pd.read_parquet("all_secret_known.parquet")
        df_prior = pd.read_parquet("all_secret_known_prior.parquet")
    except Exception as e:
        print(f"Error reading parquet files: {e}")
        return

    # Process df_ours
    df_ours['alc_floor'] = df_ours['alc'].clip(lower=-0.5)
    idx_ours = df_ours.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_ours_best = df_ours.loc[idx_ours].reset_index(drop=True)
    print(df_ours_best.columns)

    # Process df_prior
    df_prior['alc_floor'] = df_prior['alc'].clip(lower=-0.5)
    idx_prior = df_prior.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_prior_best = df_prior.loc[idx_prior].reset_index(drop=True)
    print(df_ours_best['alc_floor'].describe())
    print(df_prior_best['alc_floor'].describe())
    print("----------")
    print(df_ours_best['base_prc'].describe())
    print(df_prior_best['base_prc'].describe())

    print(f"Number of groups in 'own': {len(df_ours_best)}")
    print(f"Number of groups in 'prior': {len(df_prior_best)}")

    # Create a dataframe for common groups
    df_common = pd.merge(
        df_ours_best[['secret_column', 'known_columns', 'alc_floor']],
        df_prior_best[['secret_column', 'known_columns', 'alc_floor']],
        on=['secret_column', 'known_columns'],
        suffixes=('_ours', '_prior')
    )

    print(f"Number of common groups: {len(df_common)}")

    # Create a scatterplot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=df_common,
        x='alc_floor_prior',
        y='alc_floor_ours',
        alpha=0.7,
        edgecolor=None
    )

    # Add labels and formatting
    plt.xlabel('ALC (Prior)')
    plt.ylabel('ALC (Own)')
    plt.title('Comparison of ALC: Prior vs Own')
    plt.axline((0, 0), slope=1, color='red', linestyle='--', linewidth=1)  # Diagonal line
    plt.grid(True)

    # Save the scatterplot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prior_versus_ours.png'))
    plt.savefig(os.path.join(plots_dir, 'prior_versus_ours.pdf'))
    plt.close()


def plot_alc_best_vs_one():
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
        df_best[['secret_column', 'known_columns', 'alc_floor', 'attack_recall']],
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
    plt.figure(figsize=(6, 4))
    scatter = sns.scatterplot(
        data=df_top,
        y='alc_floor_best',
        x='alc_floor_one',
        hue='attack_recall',  # Color points by 'attack_recall'
        palette='viridis',   # Use a colormap
        edgecolor=None,
        legend=False  # Remove the legend
    )

    plt.ylabel('Worst-case ALC with recall')
    plt.xlabel('ALC without recall')
    plt.ylim(0.4, 1.05)
    plt.xlim(-0.55, 1.05)

    # Add the red shaded box (y: 0.75 to 1.0, x: -0.5 to 0)
    plt.fill_betweenx(
        y=[0.75, 1.0], x1=-0.5, x2=0, color='red', alpha=0.1, edgecolor=None
    )

    # Add the orange shaded box (y: 0.75 to 1.0, x: 0 to 0.5)
    plt.fill_betweenx(
        y=[0.75, 1.0], x1=0, x2=0.5, color='orange', alpha=0.1, edgecolor=None
    )

    # draw a diagonal line from (0,0) to (1,1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dotted', linewidth=0.5)
    # draw a horizontal line at y=0.5
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=1.0)
    plt.axhline(y=0.75, color='red', linestyle='--', linewidth=1.0)
    # draw a vertical line at x=0.5
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=1.0)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.0)
    plt.grid(True)
    # Add a colorbar directly from the scatterplot
    norm = plt.Normalize(df_top['attack_recall'].min(), df_top['attack_recall'].max())
    sm = scatter.collections[0]  # Use the scatterplot's PathCollection
    cbar = plt.colorbar(sm, label='Attack Recall', orientation='vertical', pad=0.02)

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
                print(f"Reading file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
                dir_name = os.path.dirname(file_path)
                dir_name = os.path.basename(dir_name)
                dir_name = dir_name.split('.')[0]
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

def do_config():
    jobs = []
    files_list = os.listdir(orig_files_dir)
    for file_name in files_list:
        # read in the file
        df_orig = pd.read_parquet(os.path.join(orig_files_dir, file_name))
        print(f"Get known column sets for {file_name}")
        # First populate with the cases where all columns are known
        for column in df_orig.columns:
            # make a list with all columns except column
            other_columns = [c for c in df_orig.columns if c != column]
            jobs.append({"approach": "ours", "dataset": file_name, "known_columns": other_columns})
            jobs.append({"approach": "prior", "dataset": file_name, "known_columns": other_columns})
            pass
        known_column_sets = get_good_known_column_sets(df_orig, list(df_orig.columns), max_sets=200)
        for column_set in known_column_sets:
            # make a list with all columns except column_set
            jobs.append({"approach": "ours", "dataset": file_name, "known_columns": list(column_set)})
            jobs.append({"approach": "prior", "dataset": file_name, "known_columns": list(column_set)})
    random.shuffle(jobs)
    for i, job in enumerate(jobs):
        job['job_num'] = i
    # save jobs to a json file
    with open('jobs.json', 'w') as f:
        json.dump(jobs, f, indent=4)
    slurm_script = f'''#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --output=slurm_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{len(jobs)-1}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source ../.venv/bin/activate
python compare.py attack $arrayNum
'''
    with open('slurm_script', 'w') as f:
        f.write(slurm_script)

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'attack':
            if len(sys.argv) != 3:
                print("Usage: compare.py attack <job_num>")
                quit()
            do_attack(int(sys.argv[2]))
        elif sys.argv[1] == 'plot':
            do_plots()
        elif sys.argv[1] == 'gather':
            do_gather('measure')
            do_gather('prior_measure')
        elif sys.argv[1] == 'config':
            do_config()
    else:
        print("No command line parameters were provided.")

if __name__ == "__main__":
    main()