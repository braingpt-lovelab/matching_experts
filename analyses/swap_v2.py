import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import model_utils
from utils import general_utils
from overall_accuracy_model_vs_human \
    import get_llm_accuracies as get_llm_accuracies_original
from iso_overall_accuracy_model_vs_human \
    import get_llm_accuracies as get_llm_accuracies_iso


def get_llm_accuracies(model_results_dir, use_human_abstract=True):
    llms = copy.deepcopy(model_list)
    for llm_family in llms.keys():        
        for llm in llms[llm_family]:
            if use_human_abstract:
                type_of_abstract = 'human_abstracts'
            else:
                type_of_abstract = 'llm_abstracts'

            results_dir = os.path.join(
                f"{model_results_dir}/{llm.replace('/', '--')}/"\
                f"{type_of_abstract}/swap_seed{args.seed}"
            )

            PPL_fname = "PPL_A_and_B"
            label_fname = "labels"
            PPL_A_and_B = np.load(f"{results_dir}/{PPL_fname}.npy")
            labels = np.load(f"{results_dir}/{label_fname}.npy")

            acc = general_utils.scorer_acc(PPL_A_and_B, labels)
            llms[llm_family][llm]["acc"] = acc
    return llms


def get_human_accuracies(use_human_abstract):
    """
    Overall accuracy (based on `correct` column) for `human` created cases
    """
    # Read data
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"

    correct = 0
    total = 0
    for _, row in df.iterrows():
        if row["journal_section"].startswith(who):
            correct += row["correct"]
            total += 1
    acc = correct / total
    return acc


def plot(use_human_abstract):
    """
    Plot LLMs vs human experts.

    1) Plot two bars showing average original and swapped accuracy of LLMs.
    2) Plot human experts as a horizontal line.
    """
    llms_original = get_llm_accuracies_original(model_results_dir, use_human_abstract)
    llms_iso = get_llm_accuracies_iso(model_results_dir, use_human_abstract)
    llms = get_llm_accuracies(model_results_dir, use_human_abstract)
    
    fig, ax = plt.subplots(figsize=(6, 6))

    all_llm_accuracies_original = []
    all_llm_accuracies_iso = []
    all_llm_accuracies = []
    all_llm_names = []

    for family_index, llm_family in enumerate(llms.keys()):
        for llm in llms[llm_family]:
            all_llm_accuracies_original.append(llms_original[llm_family][llm]["acc"])
            all_llm_accuracies_iso.append(llms_iso[llm_family][llm]["acc"])
            all_llm_accuracies.append(llms[llm_family][llm]["acc"])
            all_llm_names.append(llms[llm_family][llm]["llm"])

    # Plot individual LLMs original, iso and swapped accuracy
    # connected by a dotted line
    x = [0, 0.3, 0.5, 0.7, 1]
    for i in range(len(all_llm_accuracies_original)):
        ax.plot(
            x[1:3],
            [all_llm_accuracies_original[i], 
             all_llm_accuracies[i], 
            #  all_llm_accuracies_iso[i]
            ],
            color='grey',
            alpha=0.5,
            linestyle='--',
        )

    # Plot individual LLMs original, iso and swapped accuracy as curve plot
    # convert all_llm_accuracies_original, all_llm_accuracies_iso and all_llm_accuracies to pairs
    acc_pairs = list(zip(all_llm_accuracies_original, all_llm_accuracies_iso, all_llm_accuracies))
    for i, pair in enumerate(acc_pairs):
        ax.scatter(
            x,
            [0, pair[0], 0, 0, 0],
            color='purple',
            marker='*',
        )

        ax.scatter(
            x,
            [0,  0, 0, pair[1], 0],
            color='none',
            edgecolors='red',
            marker='o',
        )

        ax.scatter(
            x,
            [0, 0, pair[2], 0, 0],
            color='green',
            marker='o',
        )
    
    # Hack to get legend
    ax.scatter(
        x,
        [0, pair[0], 0, 0, 0],
        color='purple',
        marker='*',
        label="Coherent context"
    )

    ax.scatter(
        x,
        [0, 0, pair[2], 0, 0],
        color='green',
        marker='o',
        label="Swapped context"
    )

    ax.scatter(
        x,
        [0, 0, 0, pair[1], 0],
        c='none',
        edgecolors='red',
        marker='o',
        label="Without context"
    )
    
    # Plot human expert
    human_acc = get_human_accuracies(use_human_abstract)
    ax.axhline(y=human_acc, color='b', linestyle='--', linewidth=3)

    # Add annotations (Human expert)
    # In the middle of the plot, below the horizontal line
    ax.text(
        1.1,
        human_acc-0.015,
        "Human\nexperts",
        fontsize=16,
        color='k'
    )

    ax.set_ylim([0.5, 1])
    ax.set_xticks([])
    ax.set_ylabel("Accuracy")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if use_human_abstract:
        plt.savefig(f"{figs_dir}/{base_fname}_human_abstract.pdf")
    else:
        plt.savefig(f"{figs_dir}/{base_fname}_llm_abstract.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=general_utils.str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_results_dir = "model_results"
    human_results_dir = "human_results"
    figs_dir = "figs"
    base_fname = f"swap_seed{args.seed}_overall_accuracy_model_vs_human"
    model_list = model_utils.model_list
    plot(parser.parse_args().use_human_abstract)