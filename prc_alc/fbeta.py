import numpy as np
import matplotlib.pyplot as plt


def fbeta(p, r, beta):
    """
    Computes the F-beta score given precision (p), recall (r), and beta.
    
    Parameters:
        p (float): Precision
        r (float): Recall
        beta (float): Weight of recall in the F-beta score

    Returns:
        float: F-beta score
    """
    if p == 0 or r == 0:
        return 0.0
    beta_squared = beta ** 2
    return ((1 + beta_squared) * p * r) / ((beta_squared * p) + r)

def prec_from_fbeta(r, beta, fbeta_score):
    """
    Computes the precision that would produce the given F-beta score, given recall (r) and beta.
    
    Parameters:
        r (float): Recall
        beta (float): Weight of recall in the F-beta score
        fbeta_score (float): Desired F-beta score

    Returns:
        float: Precision
    """
    if r == 0 or fbeta_score == 0:
        return 0.0
    beta_squared = beta ** 2
    return (fbeta_score * r) / (((1+beta_squared) * r) - (fbeta_score * beta_squared))

def plot_prec_from_fbeta_r():
    fbeta_values = [0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # reverse the fbeta_values for plotting
    fbeta_values = fbeta_values[::-1]
    betas = [0.005, 0.01, 0.05, 0.1]
    ranges = [[0.0001, 0.00011], [0.00011, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    recall_base_values = np.concatenate(arrays)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    axes = axes.flatten()

    for idx, beta in enumerate(betas):
        ax = axes[idx]
        for fbeta in fbeta_values:
            prec_values = [prec_from_fbeta(recall_value, beta, fbeta) for recall_value in recall_base_values]
            prec_recall_pairs = [(prec, recall) for prec, recall in zip(prec_values, recall_base_values)]
            # Remove points where precision is outside [0, 1]
            prec_recall_pairs = [(prec, recall) for prec, recall in prec_recall_pairs if 0 <= prec <= 1]
            prec_recall_pairs = sorted(prec_recall_pairs, key=lambda x: x[0])
            prec_values, recall_values = zip(*prec_recall_pairs)
            ax.plot(recall_values, prec_values, label=f'FBeta = {fbeta}', linewidth=2)
        ax.set_xscale('log')
        ax.grid(True)
        ax.set_xlim(0.00009, 1)  # Set x-axis range
        ax.set_ylim(-0.05, 1.05)        # Set y-axis range
        ax.set_title(f'Beta = {beta}', fontsize=10)
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.legend(loc='lower left', fontsize='small')

    plt.tight_layout()
    plt.savefig('plots/prec_from_fbeta_r.png', dpi=300)
    plt.savefig('plots/prec_from_fbeta_r.pdf', dpi=300)
    plt.close()

# Tests
if __name__ == "__main__":
    # Note that the test case fails if p or r is 0, but not both
    test_cases = [
        {"p": 0.8, "r": 0.6, "beta": 1.0},
        {"p": 0.5, "r": 0.5, "beta": 2.0},
        {"p": 0.9, "r": 0.7, "beta": 0.5},
        {"p": 0.0, "r": 0.0, "beta": 1.0},
        {"p": 0.7, "r": 0.1, "beta": 1.0},
    ]

    for i, case in enumerate(test_cases, 1):
        p = case["p"]
        r = case["r"]
        beta = case["beta"]

        # Compute F-beta score
        fb = fbeta(p, r, beta)

        # Compute precision from F-beta score
        computed_p = prec_from_fbeta(r, beta, fb)

        # Check if the computed precision matches the original precision
        print(f"Test Case {i}:")
        print(f"  Input Precision: {p}, Recall: {r}, Beta: {beta}")
        print(f"  F-beta Score: {fb}")
        print(f"  Computed Precision: {computed_p}")
        assert abs(p - computed_p) < 1e-6, f"Test Case {i} Failed: Expected {p}, Got {computed_p}"
        print(f"  Test Case {i} Passed!\n")
    plot_prec_from_fbeta_r()