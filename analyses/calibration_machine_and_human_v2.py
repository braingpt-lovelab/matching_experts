import argparse
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from torch.nn import functional as F

from utils import model_utils
from utils import general_utils

plt.rcParams.update({"font.size": 16, "font.weight": "bold"})


def _acc(PPL_A_and_B, labels):
    """
    Given samples' PPL_A and PPL_B, and labels, compute accuracy.
    """
    pred_labels = np.ones(PPL_A_and_B.shape[0], dtype=np.int32)
    for row_index, (ppl_A, ppl_B) in enumerate(PPL_A_and_B):
        if ppl_A < ppl_B:
            pred_labels[row_index] = 0
        elif ppl_A > ppl_B:
            pred_labels[row_index] = 1
        else:
            pred_labels[row_index] = -1

    # empty bin
    if len(labels) == 0:
        return 0.

    # Consider ties as wrong
    acc = np.sum(pred_labels == labels) / (PPL_A_and_B.shape[0])
    return acc


def _plot_calibration_machine(PPL_A_and_B, labels, llm, llm_family, ax):
    """
    Plotting utility when metric is `calibration`.
    """
    # Style
    color = llms[llm_family][llm]["color"]
    alpha = llms[llm_family][llm]["alpha"]
    hatch = llms[llm_family][llm]["hatch"]
    llm = llms[llm_family][llm]["llm"]

    P_A_and_B = F.softmax(torch.tensor(PPL_A_and_B), dim=1)
    P_A_and_B_max = P_A_and_B.max(dim=1).values.numpy()

    bin_boundaries = np.linspace(0.5, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_heights = []  # acc in each bin
    overall_acc = []  # sanity check
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(
            P_A_and_B_max >= bin_lower.item(), 
            P_A_and_B_max < bin_upper.item()
        )
        prop_in_bin = in_bin.astype(float).mean()
        assert prop_in_bin == (np.sum(in_bin) / len(in_bin))

        labels_in_bin = labels[in_bin]
        samples_in_bin = PPL_A_and_B[in_bin]
        acc_in_bin = _acc(samples_in_bin, labels_in_bin)
        bin_heights.append(acc_in_bin)
        overall_acc.append(acc_in_bin * prop_in_bin)
    
    print(f"[Check] Overall Accuracy: {np.sum(overall_acc)}")

    # Plot bins as bar chart using bin_heights individually to adjust hatch
    bin_midpoints = bin_lowers + (bin_uppers - bin_lowers) / 2
    bin_widths = (bin_uppers - bin_lowers)

    for midpoint, height, width in zip(bin_midpoints, bin_heights, bin_widths):
        if hatch:
            bar = ax.bar(
                midpoint, 
                height,
                width=width, 
                edgecolor='k',
                color=color, 
                alpha=alpha,
                hatch=hatch,
            )
            # Apply the hatch offset to each bar individually
            for rectangle in bar.patches:
                rectangle.set_hatch(hatch * (5 + 1))  # Adjust the hatch pattern based on the offset
            # hatch_offset += 1 # Increment the hatch offset for the next bar
        else:
            bar = ax.bar(
                midpoint, 
                height,
                width=width, 
                edgecolor='k',
                color=color, 
                alpha=alpha,
            )

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Top answer proba.")
    ax.set_xticks([0.5, 1])
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if "Falcon-180B" in llm:
        ax.set_title(f"Falcon-180B (chat)")
    else:
        ax.set_title(f"{llm}")

    # Plot regression line
    x = np.array(bin_midpoints)
    y = np.array(bin_heights)

    # non_empty_bins = np.where(np.array(bin_heights) > 0)[0]
    # x = x[non_empty_bins]
    # y = y[non_empty_bins]

    # slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)

    # print(y)
    # print(slope)
    
    # ax.plot(
    #     x, 
    #     intercept + slope * x, 
    #     color='k', 
    #     alpha=1, 
    #     linestyle='-',
    #     linewidth=plt.rcParams['lines.linewidth'] * 2,
    # )
    return ax


def _plot_calibration_human(human_results_dir, ax):
    # Read data
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")

    n_bins = 20
    # Iterate over rows based on who created the case
    # For each who, collect confidence for correct and incorrect responses
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"
        
    confidences = []
    corrects_n_incorrects = []  # 1 and 0
    for _, row in df.iterrows():
        if row["journal_section"].startswith(who):
            # get confidence and correct
            confidence = row["confidence"]
            correct = row["correct"]
            confidences.append(confidence)
            corrects_n_incorrects.append(correct)

    # Plot calibration
    confidences = stats.rankdata(confidences, method='ordinal') - 1.
    print(confidences, len(confidences))

    # Bin the confidences and compute the accuracy per bin
    bin_boundaries = np.linspace(0, len(confidences), n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_heights = []  # acc in each bin
    overall_acc = []  # sanity check
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(
            confidences >= bin_lower.item(), 
            confidences < bin_upper.item()
        )
        prop_in_bin = in_bin.astype(float).mean()
        assert prop_in_bin == (np.sum(in_bin) / len(in_bin))

        acc_in_bin = np.mean(np.array(corrects_n_incorrects)[in_bin])
        bin_heights.append(acc_in_bin)
        overall_acc.append(acc_in_bin * prop_in_bin)
    
    print(f"[Check] Overall Accuracy: {np.sum(overall_acc)}")
    
    # Plot bins as bar chart using bin_heights
    bin_midpoints = bin_lowers + (bin_uppers - bin_lowers) / 2
    bin_widths = (bin_uppers - bin_lowers)
    ax.bar(
        bin_midpoints, 
        bin_heights,
        width=bin_widths, 
        edgecolor='k', 
        color="blue", 
        alpha=0.3, 
        # hatch=hatch
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f"Human experts")
    ax.set_xlabel("Confidence")
    ax.set_xticks([])

    # Add a regression line (fitting rank and accuracy in bin)
    x = np.array(bin_midpoints)
    y = np.array(bin_heights)
    # fit a regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # plot the regression line
    ax.plot(
        x, intercept + slope*x, 
        'k', 
        label='fitted line',
        linewidth=plt.rcParams['lines.linewidth'] * 2,
    )


def plot_calibration_human_and_machines():
    """
    4*4 figure, with the first subplot for human experts, and the rest for LLMs.
    """
    total_n_llms = 0
    for llm_family in llms:
        total_n_llms += len(llms[llm_family])
    n_cols = 4
    n_rows = (total_n_llms + 1) // n_cols  # +1 for the human experts subplot
    if (total_n_llms + 1) % n_cols > 0:  # Check if an extra row is needed
        n_rows += 1
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(12, 15), 
        sharey=True
    )

    # Plot human experts first at position (0, 0)
    ax = axes[0][0]
    _plot_calibration_human(human_results_dir, ax)

    # Adjust the loop to start filling from the second subplot
    i = 1  # Start from the second subplot
    for llm_family in llms:
        for llm in llms[llm_family]:
            results_dir = f"{model_results_dir}/{llm.replace('/', '--')}/{type_of_abstract}"
            PPL_A_and_B = np.load(f"{results_dir}/{PPL_fname}.npy")
            labels = np.load(f"{results_dir}/{label_fname}.npy")

            # Calculate subplot position
            ax_row = i // n_cols
            ax_col = i % n_cols
            ax = axes[ax_row][ax_col]
            _plot_calibration_machine(PPL_A_and_B, labels, llm, llm_family, ax)
            
            if ax_row != n_rows - 1:
                ax.set_xticks([])
            
            i += 1
                
    plt.tight_layout()
    plt.savefig(f"figs/calibration_{type_of_abstract}_v2.pdf")
    plt.close()
        

def logistic_regression_calibration(n_folds, pct_train):
    """
    Train LR on x: abs(PPL_A - PPL_B), y: correct or not for each testcase
    """
    np.random.seed(42)
    # model logisitic regression
    # using ppl diff to predict correct or not
    for llm_family in llms:
        for llm in llms[llm_family]:
            results_dir = f"{model_results_dir}/{llm.replace('/', '--')}/{type_of_abstract}"
            PPL_A_and_B = np.load(f"{results_dir}/{PPL_fname}.npy")
            labels = np.load(f"{results_dir}/{label_fname}.npy")

            # Calculate x: abs(PPL_A - PPL_B), y: correct or not
            x = np.abs(PPL_A_and_B[:, 0] - PPL_A_and_B[:, 1])
            y = []
            for abstract_idx, (ppl_A, ppl_B) in enumerate(PPL_A_and_B):
                if (
                    labels[abstract_idx] == 0 and ppl_A < ppl_B
                ) or (
                    labels[abstract_idx] == 1 and ppl_A > ppl_B
                ):
                    y.append(1)
                else:
                    y.append(0)
            
            # Create kfold split, fit the model, and produce coef and 
            # intercept as well as their standard errors; and then do significance
            # testing on coef
            coefs = []
            intercepts = []
            for fold in range(n_folds):
                # For each fold, randomly allocate `pct_train`% of the data to train
                # and the rest to test
                n_train = int(pct_train * len(x))
                train_indices = np.random.choice(len(x), n_train, replace=False)
                test_indices = np.setdiff1d(np.arange(len(x)), train_indices)
                x_train, y_train = x[train_indices], np.array(y)[train_indices]
                x_test, y_test = x[test_indices], np.array(y)[test_indices]

                # Fit LR
                lr = LogisticRegression(random_state=42)
                lr.fit(x_train.reshape(-1, 1), y_train)
                coefs.append(lr.coef_[0][0])
                intercepts.append(lr.intercept_[0])
            
            # Print the mean and standard error of the coef and intercept
            mean_coef = np.mean(coefs)
            sem_coef = stats.sem(coefs)
            mean_intercept = np.mean(intercepts)
            sem_intercept = stats.sem(intercepts)
            t, p = stats.ttest_1samp(coefs, 0)
            print(f"\nLLM: {llms[llm_family][llm]['llm']}")
            print(f"coef: {mean_coef:.2f}+-({sem_coef:.2f})")
            print(f"intercept: {mean_intercept:.2f}+-({sem_intercept:.2f})")
            display_p = f"={p/2:.3f}"
            print(f"t({n_folds-1})={t:.2f}, p(one-sided){display_p}")


    # Human experts
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"
    confidences = []
    corrects_n_incorrects = []  # 1 and 0
    for _, row in df.iterrows():
        if row["journal_section"].startswith(who):
            # get confidence and correct
            confidences.append(row["confidence"])
            corrects_n_incorrects.append(row["correct"])
    
    # Fit LR
    lr = LogisticRegression(random_state=42)
    # Create kfold split, fit the model, and produce coef and 
    # intercept as well as their standard errors; and then do significance
    # testing on coef
    coefs = []
    intercepts = []
    for fold in range(n_folds):
        # For each fold, randomly allocate `pct_train`% of the data to train
        # and the rest to test
        n_train = int(pct_train * len(confidences))
        train_indices = np.random.choice(len(confidences), n_train, replace=False)
        test_indices = np.setdiff1d(np.arange(len(confidences)), train_indices)
        x_train, y_train = np.array(confidences)[train_indices], np.array(corrects_n_incorrects)[train_indices]
        x_test, y_test = np.array(confidences)[test_indices], np.array(corrects_n_incorrects)[test_indices]

        # Fit LR
        lr.fit(x_train.reshape(-1, 1), y_train)
        coefs.append(lr.coef_[0][0])
        intercepts.append(lr.intercept_[0])
    
    # Print the mean and standard error of the coef and intercept
    mean_coef = np.mean(coefs)
    sem_coef = stats.sem(coefs)
    mean_intercept = np.mean(intercepts)
    sem_intercept = stats.sem(intercepts)
    t, p = stats.ttest_1samp(coefs, 0)
    print(f"\nHuman experts")
    print(f"coef: {mean_coef:.2f}+-({sem_coef:.2f})")
    print(f"intercept: {mean_intercept:.2f}+-({sem_intercept:.2f})")
    
    if p/2 < 0.001:
        display_p = "<.001"
    print(f"t({n_folds-1})={t:.2f}, p(one-sided){display_p}")


def main():
    # plot_calibration_human_and_machines()
    logistic_regression_calibration(n_folds=5, pct_train=0.8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_human_abstract", 
        type=general_utils.str2bool, 
        default=True
    )
    
    PPL_fname = "PPL_A_and_B"
    label_fname = "labels"
    n_bins = 10  # control % points a bin if xaxis is probability
    use_human_abstract = parser.parse_args().use_human_abstract
    if use_human_abstract:
        type_of_abstract = 'human_abstracts'
    else:
        type_of_abstract = 'llm_abstracts'
    
    model_results_dir = "model_results"
    human_results_dir = "human_results"
    llms = model_utils.model_list
    main()