import argparse
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
anon_files_dir = os.path.join('..', 'strong_data_parquet')
weak_files_dir = os.path.join('..', 'weak_data_parquet')
plots_dir = os.path.join('plots')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs('slurm_out', exist_ok=True)


def do_attack(job_num, strength):
    # read in jobs.json
    with open('jobs.json', 'r') as f:
        jobs = json.load(f)
    # get the job from the jobs list
    if job_num >= len(jobs):
        print(f"Job number {job_num} is out of range. There are only {len(jobs)} jobs.")
        sys.exit()
    job = jobs[job_num]
    print(f"Job number {job_num} started for strength {strength}.")
    pp.pprint(job)
    df_orig = pd.read_parquet(os.path.join(orig_files_dir, job['dataset']))
    if strength == 'weak':
        print(f"Reading {weak_files_dir} for anonymous data")
        df_anon = pd.read_parquet(os.path.join(weak_files_dir, job['dataset']))
    else:
        print(f"Reading {anon_files_dir} for anonymized data")
        df_anon = pd.read_parquet(os.path.join(anon_files_dir, job['dataset']))
    if job['approach'] == 'ours':
        work_files_dir = os.path.join(f'work_files_{strength}')
        use_anon_for_baseline = False
    else:
        work_files_dir = os.path.join(f'work_files_prior_{strength}')
        use_anon_for_baseline = True
    os.makedirs(work_files_dir, exist_ok=True)
    # read in the corresponding anonymized file 
    # strip suffix .parquet from file_name
    file_name = job['dataset'].split('.')[0]
    attack_dir_name = f"{file_name}.{job_num}"
    my_work_files_dir = os.path.join(work_files_dir, attack_dir_name)
    test_file_path = os.path.join(my_work_files_dir, 'summary_secret_known.csv')
    if os.path.exists(test_file_path):
        print(f"File {test_file_path} already exists. Skipping this job.")
        return
    os.makedirs(my_work_files_dir, exist_ok=True)

    brm = BrmAttack(df_original=df_orig,
                    anon=df_anon,
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

def do_plots(strength):
    # Read the parquet files into dataframes
    try:
        df_ours = pd.read_parquet(f"all_secret_known_{strength}.parquet")
        df_prior = pd.read_parquet(f"all_secret_known_prior_{strength}.parquet")
    except Exception as e:
        print(f"Error reading parquet files: {e}")
        return

    plot_prior_versus_ours(df_ours, df_prior, strength)
    plot_alc_best_vs_one(df_ours, strength)

def plot_prior_versus_ours(df_ours, df_prior, strength):
    # Process df_ours
    print(f"plot_prior_versus_ours: {strength}")
    df_ours['alc_floor'] = df_ours['alc'].clip(lower=-0.2)
    idx_ours = df_ours.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_ours_best = df_ours.loc[idx_ours].reset_index(drop=True)
    print(df_ours_best.columns)

    # Process df_prior
    df_prior['alc_floor'] = df_prior['alc'].clip(lower=-0.2)
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
        df_ours_best[['secret_column', 'known_columns', 'alc_floor', 'base_prc']],
        df_prior_best[['secret_column', 'known_columns', 'alc_floor', 'base_prc']],
        on=['secret_column', 'known_columns'],
        suffixes=('_ours', '_prior')
    )

    print(f"Number of common groups: {len(df_common)}") 

    print(f"Strength: {strength}")
    # Get the number of rows where alc_floor_ours < 0.5 and alc_floor_prior > 0.5
    alc_ours_better = df_common[(df_common['alc_floor_ours'] < 0.5) & (df_common['alc_floor_prior'] > 0.5)]
    print(f"Number of rows where ALC (ours) < 0.5 and ALC (prior) > 0.5: {len(alc_ours_better)}")
    alc_prior_better = df_common[(df_common['alc_floor_ours'] > 0.5) & (df_common['alc_floor_prior'] < 0.5)]
    print(f"Number of rows where ALC (prior) < 0.5 and ALC (ours) > 0.5: {len(alc_prior_better)}")

    alc_ours_much_better = df_common[(df_common['alc_floor_ours'] < 0.5) & (df_common['alc_floor_prior'] > 0.75)]
    print(f"Number of rows where ALC (ours) < 0.5 and ALC (prior) > 0.75: {len(alc_ours_much_better)}")
    alc_prior_much_better = df_common[(df_common['alc_floor_ours'] > 0.75) & (df_common['alc_floor_prior'] < 0.5)]
    print(f"Number of rows where ALC (prior) < 0.5 and ALC (ours) > 0.75: {len(alc_prior_much_better)}")

    df_common['prc_diff'] = df_common['base_prc_ours'] - df_common['base_prc_prior']
    print(f"Description of base PRC difference:")
    print(df_common['prc_diff'].describe())
    df_common['alc_diff'] = df_common['alc_floor_ours'] - df_common['alc_floor_prior']
    print(f"Description of base ALC difference:")
    print(df_common['alc_diff'].describe())

    # Create a scatterplot for ALC
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df_common,
        x='alc_floor_prior',
        y='alc_floor_ours',
        alpha=0.7,
        edgecolor=None
    )
    plt.xlabel('ALC (Prior)')
    plt.ylabel('ALC (Own)')
    plt.title(f'Comparison of ALC: Prior vs Own ({strength})')
    plt.axline((0, 0), slope=1, color='red', linestyle='--', linewidth=1)  # Diagonal line
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_{strength}.pdf'))
    plt.close()

    # Create a scatterplot for Baseline PRC
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df_common,
        x='base_prc_prior',
        y='base_prc_ours',
        alpha=0.7,
        edgecolor=None
    )
    plt.xlabel('Best Base PRC (Prior)')
    plt.ylabel('Best Base PRC (Own)')
    plt.title(f'Comparison of Base PRC: Prior vs Own ({strength})')
    plt.axline((0, 0), slope=1, color='red', linestyle='--', linewidth=1)  # Diagonal line
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_base_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_base_{strength}.pdf'))
    plt.close()

    # Make a boxplot for ALC
    df_melted = df_common.melt(value_vars=["alc_floor_prior", "alc_floor_ours"], 
                            var_name="Test Type", 
                            value_name="ALC")

    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df_melted, x="ALC", y="Test Type", orient="h")
    plt.xlabel("ALC", fontsize=12)
    plt.ylabel("Test Type", fontsize=12)
    plt.title("Comparison of ALC: Prior vs Ours", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_box_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_box_{strength}.pdf'))
    plt.close()

    # Make a boxplot for Base PRC
    df_melted = df_common.melt(value_vars=["base_prc_prior", "base_prc_ours"], 
                            var_name="Test Type", 
                            value_name="Base PRC")

    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df_melted, x="Base PRC", y="Test Type", orient="h")
    plt.xlabel("Base PRC", fontsize=12)
    plt.ylabel("Test Type", fontsize=12)
    plt.title("Comparison of Base PRC: Prior vs Ours", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_base_box_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_base_box_{strength}.pdf'))
    plt.close()

def plot_recall_boxes():
    df_weak = pd.read_parquet(f"all_secret_known_weak.parquet")
    df_strong = pd.read_parquet(f"all_secret_known_strong.parquet")
    df_weak['alc_floor'] = df_weak['alc'].clip(lower=-0.2)
    df_strong['alc_floor'] = df_strong['alc'].clip(lower=-0.2)
    df_best_weak = df_weak[df_weak['paired'] == False].reset_index(drop=True)
    df_best_strong = df_strong[df_strong['paired'] == False].reset_index(drop=True)

    # For each value of column halt_code in df_best_strong, count the rows
    # and print the counts
    print(f"Counts of halt_code in df_best_strong:")
    print(df_best_strong['halt_code'].value_counts())
    print(f"Counts of halt_code in df_best_weak:")
    print(df_best_weak['halt_code'].value_counts())


    df_one_weak = df_weak[df_weak['attack_recall'] == 1].reset_index(drop=True)
    idx = df_one_weak.groupby(['secret_column', 'known_columns'])['alc_floor'].idxmax()
    df_one_weak = df_one_weak.loc[idx].reset_index(drop=True)
    df_one_strong = df_strong[df_strong['attack_recall'] == 1].reset_index(drop=True)
    idx = df_one_strong.groupby(['secret_column', 'known_columns'])['alc_floor'].idxmax()
    df_one_strong = df_one_strong.loc[idx].reset_index(drop=True)
    df_merged_strong = pd.merge(
        df_best_strong[['secret_column', 'known_columns', 'alc_floor']],
        df_one_strong[['secret_column', 'known_columns', 'alc_floor']],
        on=['secret_column', 'known_columns'],
        suffixes=('_best', '_one')
    )
    df_merged_weak = pd.merge(
        df_best_weak[['secret_column', 'known_columns', 'alc_floor']],
        df_one_weak[['secret_column', 'known_columns', 'alc_floor']],
        on=['secret_column', 'known_columns'],
        suffixes=('_best', '_one')
    )
    # compute the difference of alc_floor_best and alc_floor_one
    df_merged_strong['alc_difference'] = df_merged_strong['alc_floor_best'] - df_merged_strong['alc_floor_one']
    df_merged_weak['alc_difference'] = df_merged_weak['alc_floor_best'] - df_merged_weak['alc_floor_one']

    df_prc_cnt_weak = (df_weak.groupby(['secret_column', 'known_columns']).size().reset_index(name='num_prc'))
    df_prc_cnt_weak['num_prc'] -= 1
    df_prc_cnt_strong = (df_strong.groupby(['secret_column', 'known_columns']).size().reset_index(name='num_prc'))
    df_prc_cnt_strong['num_prc'] -= 1

    print(f"Describe num_prc for strong:")
    print(df_prc_cnt_strong['num_prc'].describe())
    print(f"Describe num_prc for weak:")
    print(df_prc_cnt_weak['num_prc'].describe())

    df_prc_cnt_weak_low = ( df_weak[df_weak['halt_code'] == 'extreme_low'] .groupby(['secret_column', 'known_columns']) .size() .reset_index(name='num_prc'))
    df_prc_cnt_weak_high = ( df_weak[df_weak['halt_code'] != 'extreme_low'] .groupby(['secret_column', 'known_columns']) .size() .reset_index(name='num_prc'))
    df_prc_cnt_strong_low = ( df_strong[df_strong['halt_code'] == 'extreme_low'] .groupby(['secret_column', 'known_columns']) .size() .reset_index(name='num_prc'))
    df_prc_cnt_strong_high = ( df_strong[df_strong['halt_code'] != 'extreme_low'] .groupby(['secret_column', 'known_columns']) .size() .reset_index(name='num_prc'))
    print(f"Describe num_prc for weak extreme_low only:")
    print(df_prc_cnt_weak_low['num_prc'].describe())
    print(f"Describe num_prc for weak all but extreme_low:")
    print(df_prc_cnt_weak_high['num_prc'].describe())
    print(f"Describe num_prc for strong extreme_low only:")
    print(df_prc_cnt_strong_low['num_prc'].describe())
    print(f"Describe num_prc for strong all but extreme_low:")
    print(df_prc_cnt_strong_high['num_prc'].describe())

    # Count the number of rows where alc_floor_best > 0.75 and alc_floor_one < 0.5
    factor = 100/len(df_merged_strong)
    rnd = 2
    with_safe_without_unsafe_strong = round(factor * len(df_merged_strong[(df_merged_strong['alc_floor_best'] < 0.5) & (df_merged_strong['alc_floor_one'] >= 0.75)]), rnd)
    with_unsafe_without_safe_strong = round(factor * len(df_merged_strong[(df_merged_strong['alc_floor_best'] >= 0.75) & (df_merged_strong['alc_floor_one'] < 0.5)]), rnd)
    with_risk_without_safe_strong = round(factor * len(df_merged_strong[(df_merged_strong['alc_floor_best'] >= 0.5) & (df_merged_strong['alc_floor_best'] < 0.75) & (df_merged_strong['alc_floor_one'] < 0.5)]), rnd)
    with_safe_without_risk_strong = round(factor * len(df_merged_strong[(df_merged_strong['alc_floor_one'] >= 0.5) & (df_merged_strong['alc_floor_one'] < 0.75) & (df_merged_strong['alc_floor_best'] < 0.5)]), rnd)

    with_safe_without_unsafe_weak = round(factor * len(df_merged_weak[(df_merged_weak['alc_floor_best'] < 0.5) & (df_merged_weak['alc_floor_one'] >= 0.75)]), rnd)
    with_unsafe_without_safe_weak = round(factor * len(df_merged_weak[(df_merged_weak['alc_floor_best'] >= 0.75) & (df_merged_weak['alc_floor_one'] < 0.5)]), rnd)
    with_risk_without_safe_weak = round(factor * len(df_merged_weak[(df_merged_weak['alc_floor_best'] >= 0.5) & (df_merged_weak['alc_floor_best'] < 0.75) & (df_merged_weak['alc_floor_one'] < 0.5)]), rnd)
    with_safe_without_risk_weak = round(factor * len(df_merged_weak[(df_merged_weak['alc_floor_one'] >= 0.5) & (df_merged_weak['alc_floor_one'] < 0.75) & (df_merged_weak['alc_floor_best'] < 0.5)]), rnd)

    # Let's make a nice table!
    tab = f'''
\\begin{{table}}[t]
\\begin{{center}}
\\begin{{small}}
\\begin{{tabular}}{{cc|cc}}
\\toprule
 Recall & No-recall  & Strong Anon & Weak Anon \\\\
\\midrule
At risk & Safe & {with_risk_without_safe_strong}\\% & {with_risk_without_safe_weak}\\% \\\\
Unsafe & Safe & {with_unsafe_without_safe_strong}\\% & {with_unsafe_without_safe_weak}\\% \\\\
Safe & At risk & {with_safe_without_risk_strong}\\% & {with_safe_without_risk_weak}\\% \\\\
Safe & Unsafe & {with_safe_without_unsafe_strong}\\% & {with_safe_without_unsafe_weak}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Percentage of {len(df_merged_strong)} attacks where the prior no-recall approach reaches the wrong conclusion.}}
\\label{{tab:wrong_conclusion}}
\\end{{small}}
\\end{{center}}
\\end{{table}}
'''
    # Write tab to a file
    with open('plots/wrong_conclusion.tex', 'w') as f:
        f.write(tab)

    # Prepare data for recall boxplots
    df_combined_recall = pd.DataFrame({
        "Value": (
            list(df_best_strong["base_recall"]) +
            list(df_best_strong["attack_recall"]) +
            list(df_best_weak["attack_recall"])
        ),
        "Category": (
            ["Base recall"] * len(df_best_strong["base_recall"]) +
            ["Attack recall,\nstrong anon"] * len(df_best_strong["attack_recall"]) +
            ["Attack recall,\nweak anon"] * len(df_best_weak["attack_recall"])
        )
    })

    # Prepare data for difference boxplots
    df_combined_diff = pd.DataFrame({
        "Value": (
            list(df_merged_strong["alc_difference"]) +
            list(df_merged_weak["alc_difference"])
        ),
        "Category": (
            ["ALC difference,\nstrong anon"] * len(df_merged_strong["alc_difference"]) +
            ["ALC difference,\nweak anon"] * len(df_merged_weak["alc_difference"])
        )
    })

    # Create the subplots
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.2), gridspec_kw={'height_ratios': [3, 2]}, sharex=False)

    # Plot recall boxplots
    sns.boxplot(data=df_combined_recall, x="Value", y="Category", orient="h", ax=axes[0])
    axes[0].set_xlabel("Recall (experiments with recall)", fontsize=10)
    axes[0].set_ylabel("")  # Remove y-axis label

    # Plot difference boxplots
    sns.boxplot(data=df_combined_diff, x="Value", y="Category", orient="h", ax=axes[1])
    axes[1].set_xlabel("Clipped ALC Difference (with recall - no recall)", fontsize=10)
    axes[1].set_ylabel("")  # Remove y-axis label

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("plots/recall_and_diff_boxes.png", dpi=300)
    plt.savefig("plots/recall_and_diff_boxes.pdf", dpi=300)
    plt.close()

def plot_alc_best_vs_one(df, strength):
    print(f"plot_alc_best_vs_one: {strength}")
    # make a new column called alc_floor where all alc values less than -0.5 are set to -0.5
    df['alc_floor'] = df['alc'].clip(lower=-0.2)
    df_best = df[df['paired'] == False].reset_index(drop=True)
    print(f"Number of rows in 'best': {len(df_best)}")
    num_clipped = len(df_best[df_best['alc'] < -0.2])
    print(f"Number of best clipped alc values: {num_clipped}")
    print(f"Fraction of best alc values that are clipped: {num_clipped / len(df_best)}")
    idx = df_best.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_best = df_best.loc[idx].reset_index(drop=True)
    print(f"Number of rows in 'best' after grouping: {len(df_best)}")
    df_one = df[df['attack_recall'] == 1].reset_index(drop=True)
    print(f"Number of rows in 'one': {len(df_one)}")
    idx = df_one.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_one = df_one.loc[idx].reset_index(drop=True)
    print(f"df_one after grouping: {len(df_one)}")
    num_clipped = len(df_one[df_one['alc'] < -0.2])
    print(f"Number of one clipped alc values: {num_clipped}")
    print(f"Fraction of one alc values that are clipped: {num_clipped / len(df_best)}")
    # take the sum of base_count in df_one
    print(f"Total number of predictions: {df_one['base_count'].sum()}")
    print(f"Average predictions per attack: {df_one['base_count'].sum() / len(df_one)}")
    # count the number of rows in df where both base_si and attack_si are <= 0.1, and paired is True
    df_paired = df[(df['base_si'] <= 0.1) & (df['attack_si'] <= 0.1) & (df['paired'] == True)]
    print(f"Number of significant PRC scores: {len(df_paired)}")
    print(f"Average significant PRC scores per attack: {len(df_paired) / len(df_one)}")
    # compute the average of num_known_columns in df_best
    print(f"Average number of known columns in best: {df_best['num_known_columns'].mean()}")
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
    print(f"Number of rows in merged dataframe: {len(df_merged)}")
    # Compute the difference of alc values
    df_merged['alc_difference'] = df_merged['alc_floor_best'] - df_merged['alc_floor_one']
    print("df_best alc:")
    print(df_best['alc_floor'].describe())
    print("df_one alc:")
    print(df_one['alc_floor'].describe())
    # describe alc_difference
    print(df_merged['alc_difference'].describe())

    # Count the number of rows where alc_floor_best > 0.75 and alc_floor_one < 0.5
    alc_best_much_better = df_merged[(df_merged['alc_floor_best'] > 0.75) & (df_merged['alc_floor_one'] < 0.5)]
    print(f"Number of rows where ALC (best) > 0.75 and ALC (one) < 0.5 ({strength}): {len(alc_best_much_better)}, {len(alc_best_much_better) / len(df_merged)}")
    # Count the number of rows where alc_floor_best is between 0.5 and 0.75 and alc_floor_one < 0.5
    alc_best_better = df_merged[(df_merged['alc_floor_best'] > 0.5) & (df_merged['alc_floor_best'] < 0.75) & (df_merged['alc_floor_one'] < 0.5)]
    print(f"Number of rows where ALC (best) between 0.5-0.75 and ALC (one) < 0.5 ({strength}): {len(alc_best_better)}, {len(alc_best_better) / len(df_merged)}")


    # make df_top which is df_merged where alc_floor >= 0.5
    df_top = df_merged[df_merged['alc_floor_best'] >= 0.5]
    print(f"There are {len(df_top)} rows with alc >= 0.5")
    print(f"Top group described:")
    print(df_top['alc_difference'].describe())

    df_merged = df_merged.sort_values(by='attack_recall', ascending=False)
    # make a seaborn scatterplot from df_top with alc_floor_best on x and alc_floor_one on y
    plt.figure(figsize=(5, 3))
    scatter = sns.scatterplot(
        data=df_merged,
        #data=df_top,
        y='alc_floor_best',
        x='alc_floor_one',
        hue='attack_recall',  # Color points by 'attack_recall'
        palette='viridis',   # Use a colormap
        edgecolor=None,
        s=10,
        legend=False  # Remove the legend
    )

    plt.ylabel('Clipped ALC with recall')
    plt.xlabel('Clipped ALC no recall')
    #plt.ylim(0.4, 1.05)
    #plt.xlim(-0.55, 1.05)

    # Add the shaded boxes
    plt.fill_betweenx(
        y=[0.75, 1.0], x1=0, x2=0.5, color='red', alpha=0.1, edgecolor=None
    )
    plt.fill_betweenx(
        y=[0.5, 0.75], x1=0, x2=0.5, color='orange', alpha=0.1, edgecolor=None
    )

    # draw a diagonal line from (0,0) to (1,1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dotted', linewidth=0.5)
    # draw a horizontal line at y=0.5
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0)
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=1.0)
    plt.axhline(y=0.75, color='red', linestyle='--', linewidth=1.0)
    # draw a vertical line at x=0.5
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.0)
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=1.0)
    plt.axvline(x=0.75, color='red', linestyle='--', linewidth=1.0)
    plt.grid(True)
    # Add a colorbar directly from the scatterplot
    norm = plt.Normalize(df_top['attack_recall'].min(), df_top['attack_recall'].max())
    sm = scatter.collections[0]  # Use the scatterplot's PathCollection
    cbar = plt.colorbar(sm, label='Attack Recall', orientation='vertical', pad=0.02)

    # tighten
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'alc_best_vs_one_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'alc_best_vs_one_{strength}.pdf'))
    plt.close()

    print(f"Description of base_recall for best PRC:")
    print(df_best['base_recall'].describe())
    print(f"Description of attack_recall for best PRC:")
    print(df_best['attack_recall'].describe())

    # Make a boxplot for recall
    df_melted = df_best.melt(value_vars=["base_recall", "attack_recall"], 
                            var_name="Test Type", 
                            value_name="Recall")

    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df_melted, x="Recall", y="Test Type", orient="h")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Test Type", fontsize=12)
    plt.title("Comparison of Recall: Baseline versus Attack", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'recalls_box_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'recalls_box_{strength}.pdf'))
    plt.close()

def do_gather(measure_type, strength):
    print(f"Gathering files for {measure_type}, strength {strength}...")
    work_files_dir = os.path.join(f'work_files_{strength}')
    out_name = f'all_secret_known_{strength}.parquet'
    if measure_type == 'prior_measure':
        work_files_dir = os.path.join(f'work_files_prior_{strength}')
        out_name = f'all_secret_known_prior_{strength}.parquet'
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
    with open('slurm_script_strong', 'w') as f:
        f.write(slurm_script)
    slurm_script = f'''#!/bin/bash
#SBATCH --job-name=compare_weak
#SBATCH --output=slurm_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{len(jobs)-1}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source ../.venv/bin/activate
python compare.py --weak attack $arrayNum
'''
    with open('slurm_script_weak', 'w') as f:
        f.write(slurm_script)

def main():
    parser = argparse.ArgumentParser(description="Run attacks, plots, or configuration.")
    parser.add_argument("command", choices=["attack", "plot", "gather", "config"], help="Command to execute")
    parser.add_argument("job_num", nargs="?", type=int, help="Job number (required for 'attack')")
    parser.add_argument("-w", "--weak", action="store_true", help="Use weak strength (default is strong)")

    args = parser.parse_args()

    # Determine the strength based on the -w/--weak flag
    strength = "weak" if args.weak else "strong"

    if args.command == "attack":
        if args.job_num is None:
            print("Error: 'attack' command requires a job number.")
            sys.exit(1)
        do_attack(args.job_num, strength)
    elif args.command == "plot":
        do_plots('strong')
        do_plots('weak')
        plot_recall_boxes()
    elif args.command == "gather":
        do_gather('measure', 'strong')
        do_gather('prior_measure', 'strong')
        do_gather('measure', 'weak')
        do_gather('prior_measure', 'weak')
    elif args.command == "config":
        do_config()

if __name__ == "__main__":
    main()