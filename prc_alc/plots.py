import numpy as np
import os
import matplotlib.pyplot as plt
from anonymity_loss_coefficient import AnonymityLossCoefficient
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pprint
pp = pprint.PrettyPrinter(indent=4)

plots_path = os.path.join('plots')
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

def savefigs(plt, name):
    for suffix in ['.png', '.pdf']:
        path_name = name + suffix
        out_path = os.path.join(plots_path, path_name)
        plt.savefig(out_path)

def plot_prec_recall_for_diff_rmin(out_name):
    ''' The purpose of this plot is to see how different values of
        alpha effect the prc curve
    '''
    alc = AnonymityLossCoefficient()
    if alc.get_param('prc_abs_weight') != 0.0:
        raise ValueError('prc_abs_weight is not 0.0')
    prc_val = 0.5
    alpha = alc.get_param('recall_adjust_strength')
    if alc.get_param('prc_abs_weight') != 0.0:
        raise ValueError('prc_abs_weight is not 0.0')
    rmins = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    ranges = [[0.0001, 0.00011], [0.00011, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    recall_base_values = np.concatenate(arrays)

    plt.figure(figsize=((4.5, 3.5)))
    for rmin in rmins:
        alc.set_param('recall_adjust_min_intercept', rmin)
        prec_values = [alc.prec_from_prc_recall(prc_val, recall_value) for recall_value in recall_base_values]
        prec_recall_pairs = [(prec, recall) for prec, recall in zip(prec_values, recall_base_values)]
        prec_recall_pairs = sorted(prec_recall_pairs, key=lambda x: x[0])
        prec_recall_pairs = [(prec, recall) for prec, recall in prec_recall_pairs if prec <= 1.0]
        prec_values, recall_values = zip(*prec_recall_pairs)
        plt.scatter(recall_values, prec_values, label=f'Rmin = {rmin}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(0.4, 1.05)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.text(0.05, 0.98, f'PRC = {prc_val}, alpha = {alpha}', ha='left', va='top', fontsize=9, transform=plt.gca().transAxes)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='upper right')
    plt.tight_layout()
    savefigs(plt, out_name)


def plot_prec_recall_for_diff_alpha(out_name):
    ''' The purpose of this plot is to see how different values of
        alpha effect the prc curve
    '''
    alc = AnonymityLossCoefficient()
    if alc.get_param('prc_abs_weight') != 0.0:
        raise ValueError('prc_abs_weight is not 0.0')
    prc_val = 0.5
    alphas = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    ranges = [[0.0001, 0.00011], [0.00011, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    recall_base_values = np.concatenate(arrays)

    plt.figure(figsize=((4.5, 3.5)))
    for alpha in alphas:
        alc.set_param('recall_adjust_strength', alpha)
        prec_values = [alc.prec_from_prc_recall(prc_val, recall_value) for recall_value in recall_base_values]
        prec_recall_pairs = [(prec, recall) for prec, recall in zip(prec_values, recall_base_values)]
        prec_recall_pairs = sorted(prec_recall_pairs, key=lambda x: x[0])
        prec_recall_pairs = [(prec, recall) for prec, recall in prec_recall_pairs if prec <= 1.0]
        prec_values, recall_values = zip(*prec_recall_pairs)
        plt.scatter(recall_values, prec_values, label=f'alpha = {alpha}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(0.4, 1.05)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.text(0.05, 0.98, f'PRC = {prc_val}, Rmin = 0.0001', ha='left', va='top', fontsize=9, transform=plt.gca().transAxes)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='upper right')
    plt.tight_layout()
    savefigs(plt, out_name)


def plot_prec_recall_for_equal_prc(out_name):
    ''' The purpose of this plot is to see how different values of prec
        and recall can have the same prc.
    '''
    alc = AnonymityLossCoefficient()
    alpha = alc.get_param('recall_adjust_strength')
    if alc.get_param('prc_abs_weight') != 0.0:
        raise ValueError('prc_abs_weight is not 0.0')
    print(f'for prc = 0.5, and prec 1.0, recall = {alc.recall_from_prc_prec(0.5, 1.0)}')
    print(f'for prc = 0.5, and recall 0.001484, prec = {alc.prec_from_prc_recall(0.5, 0.001484)}')
    print(f'for prc = 0.5, and prec 0.6, recall = {alc.recall_from_prc_prec(0.5, 0.6)}')
    print(f'for prc = 0.5, and recall 0.0233, prec = {alc.prec_from_prc_recall(0.5, 0.0233)}')
    print(f'for prc = 0.5, and prec 0.5, recall = {alc.recall_from_prc_prec(0.5, 0.5)}')
    print(f'for prc = 0.5, and recall 1.0, prec = {alc.prec_from_prc_recall(0.5, 1.0)}')
    prc_vals = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001]
    ranges = [[0.0001, 0.00011], [0.00011, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    recall_base_values = np.concatenate(arrays)

    plt.figure(figsize=((4.5, 3.5)))
    for prc_val in prc_vals:
        prec_values = [alc.prec_from_prc_recall(prc_val, recall_value) for recall_value in recall_base_values]
        prec_recall_pairs = [(prec, recall) for prec, recall in zip(prec_values, recall_base_values)]
        prec_recall_pairs = sorted(prec_recall_pairs, key=lambda x: x[0])
        prec_recall_pairs = [(prec, recall) for prec, recall in prec_recall_pairs if prec <= 1.0]
        prec_values, recall_values = zip(*prec_recall_pairs)
        plt.scatter(recall_values, prec_values, label=f'PRC = {prc_val}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.text(0.05, 0.98, f'alpha = {alpha}, Rmin = 0.0001', ha='left', va='top', fontsize=9, transform=plt.gca().transAxes)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='lower center')
    plt.tight_layout()
    savefigs(plt, out_name)

def run_prc_checks(alc, p_base, r_base, prr_base):
    if r_base <= 0.0001:
        return
    p_base_test = round(alc.prec_from_prc_recall(prr_base, r_base),3)
    if round(p_base,3) != p_base_test:
        print(f'Error: prec_from_prc_recall({prr_base}, {r_base})')
        print(f'Expected: {round(p_base,3)}, got: {p_base_test}')
        quit()
    r_base_test = round(alc.recall_from_prc_prec(prr_base, p_base),3)
    if round(r_base,3) != r_base_test:
        print(f'Error: recall_from_prc_prec({prr_base}, {p_base})')
        print(f'Expected: {round(r_base,3)}, got: {r_base_test}')
        quit()

def do_alc_test(alc, p_base, r_base, increase, r_attack):
    print('------------------------------------')
    p_attack = p_base + increase * (1.0 - p_base)
    print(f'Base precision: {p_base}, base recall: {r_base}\nattack precision: {p_attack}, attack recall: {r_attack}')
    prc_atk = alc.prc(prec=p_attack, recall=r_attack)
    print(f'prc_atk: {prc_atk}')
    prr_base = alc.prc(prec=p_base, recall=r_base)
    print(f'prr_base: {prr_base}')
    run_prc_checks(alc, p_base, r_base, prr_base)
    print(f'ALC: {round(alc.alc(p_base=p_base, r_base=r_base, p_attack=p_attack, r_attack=r_attack),3)}')

def make_alc_plots(recall_adjust_strength=3.0, pairs='v3'):
    alc = AnonymityLossCoefficient()
    if alc.get_param('prc_abs_weight') != 0.0:
        raise ValueError('prc_abs_weight is not 0.0')
    alc.set_param('recall_adjust_strength', recall_adjust_strength)
    if pairs == 'v1':
        Ratk_Rbase_pairs = [(1, 1), (0.01, 0.01), (0.7, 1.0), (0.01, 0.05)]
        fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    elif pairs == 'v2':
        Ratk_Rbase_pairs = [(1, 1), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001)]
        fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    elif pairs == 'v3':
        Ratk_Rbase_pairs = [(1, 1), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001), (0.05, 0.1), (0.025, 0.1)]
        fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    else:
        raise ValueError(f'Invalid pairs version: {pairs}')
    Pbase_values = [0.01, 0.1, 0.4, 0.7, 0.9]
    Patk = np.arange(0, 1.01, 0.01)
    
    axs = axs.flatten()
    
    for i, (Ratk, Rbase) in enumerate(Ratk_Rbase_pairs):
        for Pbase in Pbase_values:
            ALC = [alc.alc(p_base=Pbase, r_base=Rbase, p_attack=p, r_attack=Ratk) for p in Patk]
            axs[i].plot(Patk, ALC, label=f'Pbase={Pbase}')
        
        axs[i].text(0.05, 0.95, f'Ratk = {Ratk}, Rbase = {Rbase}\nalpha = {recall_adjust_strength}\nRmin = 0.0001', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(-0.5, 1)
        
        # Remove x-axis labels and ticks for the upper two subplots
        if i < len(Ratk_Rbase_pairs) - 2:
            axs[i].set_xlabel('')
            #axs[i].set_xticklabels([])
        
        # Remove y-axis labels and ticks for the right subplots
        if i % 2 == 1:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])
        
        if i % 2 == 0:
            axs[i].set_ylabel('ALC')
        
        if i >= len(Ratk_Rbase_pairs) - 2:
            axs[i].set_xlabel('Patk')
        
        axs[i].legend(fontsize='small', loc='lower right')
        axs[i].grid(True)
    
    plt.tight_layout()
    
    # Save the plot in both PNG and PDF formats
    savefigs(plt, f'alc_plot_{recall_adjust_strength}_{pairs}.png')
    savefigs(plt, f'alc_plot_{recall_adjust_strength}_{pairs}.pdf')


def plot_basic_alc(alc):
    # Define the range for prr_base
    prr_base_values = np.linspace(0, 1.0, 100)

    # Define the alc values
    alc_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the curves for each alc value
    for alc_val in alc_values:
        prc_atk_values = [alc.prcatk_from_prcbase_alc(prr_base, alc_val) for prr_base in prr_base_values]
        filtered_prr_base_values = [prr_base for prr_base, prc_atk in zip(prr_base_values, prc_atk_values) if prc_atk < 1.0]
        filtered_prc_atk_values = [prc_atk for prc_atk in prc_atk_values if prc_atk < 1.0]
        ax.plot(filtered_prr_base_values, filtered_prc_atk_values, label=f'ALC = {alc_val}', linewidth=3)

    # Set the labels with larger font size
    ax.set_xlabel('PRC Base', fontsize=16)
    ax.set_ylabel('PRC Attack', fontsize=18)

    # Set the tick parameters with larger font size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add a legend
    ax.legend()
    ax.grid(True)

    # Create an inset plot
    prr_base_values = np.linspace(0.95, 1.0, 100)
    ax_inset = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.48, 0.15, 0.30, 0.30), bbox_transform=ax.transAxes, loc='upper left')
    for alc_val in alc_values:
        prc_atk_values = [alc.prcatk_from_prcbase_alc(prr_base, alc_val) for prr_base in prr_base_values]
        filtered_prr_base_values = [prr_base for prr_base, prc_atk in zip(prr_base_values, prc_atk_values) if prc_atk < 1.0]
        filtered_prc_atk_values = [prc_atk for prc_atk in prc_atk_values if prc_atk < 1.0]
        ax_inset.plot(filtered_prr_base_values, filtered_prc_atk_values, label=f'ALC = {alc_val}', linewidth=3)

    # Set the tick parameters for the inset plot
    ax_inset.tick_params(axis='both', which='major', labelsize=10)

    # Remove the legend from the inset plot
    ax_inset.legend().set_visible(False)

    # Show the plot
    plt.tight_layout()
    return plt


def p_atk_from_p_base_recall(alc, p_base, recall, alc_val=0.5):
    prr_base = alc.prc(p_base, recall)
    prc_atk = alc.prcatk_from_prcbase_alc(prr_base, alc_val)
    p_atk = alc.prec_from_prc_recall(prc_atk, recall)
    return p_atk if p_atk <= 1.0 else None

def plot_recall_alc(alc):
    equal_alc = 0.5
    # Define the range for prr_base
    prer_base_values = np.linspace(0, 1.0, 100)

    recall_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the curves for each alc value
    for recall in recall_values:
        p_atk_values = [p_atk_from_p_base_recall(alc, p_base, recall, alc_val=equal_alc) for p_base in prer_base_values]
        filtered_p_base_values = [p_base for p_base, p_atk in zip(prer_base_values, p_atk_values) if p_atk is not None]
        filtered_p_atk_values = [p_atk for p_atk in p_atk_values if p_atk is not None]
        ax.plot(filtered_p_base_values, filtered_p_atk_values, label=f'Recall = {recall}', linewidth=3)

    # Set the labels with larger font size
    ax.set_xlabel('Precision Base', fontsize=16)
    ax.set_ylabel('Precision Attack', fontsize=18)

    # Set the tick parameters with larger font size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add a legend
    ax.legend()
    ax.grid(True)
    plt.text(0.05, 0.98, f'All ALC = {equal_alc}', ha='left', va='top', fontsize=10, transform=plt.gca().transAxes)

    plt.tight_layout()
    return plt


def generate_points(alc, c_ratio, p_base, num_points=1000):
    points = []
    for _ in range(num_points):
        p_attack = p_base
        r_attack = np.random.uniform(0.001, 1.0)
        r_base = r_attack * c_ratio
        alc_val = alc.alc(p_base=p_base, r_base=r_base, p_attack=p_attack, r_attack=r_attack)
        points.append((r_attack, alc_val))
    return points

def plot_alc_scatter(alc):
    c_ratios = [1/1.1, 0.5, 0.1, 0.05]
    colors = ['red', 'blue', 'green', 'orange']
    #labels = [f'r_base/r_attack = {c_ratio}' for c_ratio in c_ratios]
    labels = [f'r_attack/r_base = {1/c_ratio}' for c_ratio in c_ratios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate points and plot for p_base = 0.5
    point_size = 10
    p_base = 0.5
    all_alc_values_1 = []
    for c_ratio, color, label in zip(c_ratios, colors, labels):
        points = generate_points(alc, c_ratio, p_base)
        r_attack_values, alc_values = zip(*points)
        all_alc_values_1.extend(alc_values)
        ax1.scatter(r_attack_values, alc_values, color=color, label=label, alpha=0.6, s=point_size)
    ax1.set_xlabel('Attack Recall', fontsize=14)
    ax1.set_ylabel('ALC', fontsize=14)
    ax1.set_title(f'Baseline and Attack Precision = {p_base}', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Generate points and plot for p_base = 0.9
    p_base = 0.95
    all_alc_values_2 = []
    for c_ratio, color, label in zip(c_ratios, colors, labels):
        points = generate_points(alc, c_ratio, p_base)
        r_attack_values, alc_values = zip(*points)
        all_alc_values_2.extend(alc_values)
        ax2.scatter(r_attack_values, alc_values, color=color, label=label, alpha=0.6, s=point_size)
    ax2.set_xlabel('Attack Recall', fontsize=14)
    ax2.set_ylabel('ALC', fontsize=14)
    ax2.set_title(f'Baseline and Attack Precision = {p_base}', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    # Determine the combined y-axis range
    combined_y_min = min(min(all_alc_values_1), min(all_alc_values_2))
    combined_y_max = max(max(all_alc_values_1), max(all_alc_values_2))

    # Set the same y-axis range for both plots
    ax1.set_ylim(combined_y_min, combined_y_max+0.05)
    ax2.set_ylim(combined_y_min, combined_y_max+0.05)

    plt.tight_layout()
    return plt

alc = AnonymityLossCoefficient()
do_alc_test(alc, p_base=0.5, r_base=1.0, increase=0.2, r_attack=1.0)
do_alc_test(alc, p_base=0.2, r_base=1.0, increase=0.8, r_attack=1.0)
do_alc_test(alc, p_base=0.999, r_base=1.0, increase=0.9, r_attack=1.0)
do_alc_test(alc, p_base=0.5, r_base=0.1, increase=0.2, r_attack=0.1)
do_alc_test(alc, p_base=0.2, r_base=0.1, increase=0.8, r_attack=0.1)
do_alc_test(alc, p_base=0.5, r_base=0.01, increase=0.2, r_attack=0.01)
do_alc_test(alc, p_base=0.2, r_base=0.01, increase=0.8, r_attack=0.01)
do_alc_test(alc, p_base=0.5, r_base=0.001, increase=0.2, r_attack=0.001)
do_alc_test(alc, p_base=0.2, r_base=0.001, increase=0.8, r_attack=0.001)
do_alc_test(alc, p_base=0.5, r_base=0.0001, increase=0.2, r_attack=0.0001)
do_alc_test(alc, p_base=0.2, r_base=0.0001, increase=0.8, r_attack=0.0001)
do_alc_test(alc, p_base=1.0, r_base=0.00001, increase=0, r_attack=0.00001)
plot_prec_recall_for_diff_rmin('prec_recall_for_diff_rmin')
plot_prec_recall_for_diff_alpha('prec_recall_for_diff_alpha')
plot_prec_recall_for_equal_prc('prec_recall_for_equal_prc')
plt = plot_alc_scatter(alc)
savefigs(plt, 'recall_mismatch')
plt = plot_recall_alc(alc)
savefigs(plt, 'alc_recall')
plt = plot_basic_alc(alc)
savefigs(plt, 'alc_basic')
make_alc_plots(pairs='v3')
for recall_adjust_strength in [1.0, 2.0, 3.0, 4.0]:
    make_alc_plots(recall_adjust_strength=recall_adjust_strength, pairs='v1')
    make_alc_plots(recall_adjust_strength=recall_adjust_strength, pairs='v2')
