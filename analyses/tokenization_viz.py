import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from decouple import config as env_config

"""
Compare tokenization of the same BrainBench abstract using different tokenizers.

We use abstract that 
    1. GPT-2 uses the pretrained tokenizer got incorrect
    2. GPT-2 uses the neuro-tokenizer got correct
"""

def _load_abstract():
    df = pd.read_csv(abstracts_fpath)
    abstracts = df["original_abstract"]
    return abstracts


def _locate_examples(llm_1, llm_2, abstract_idx):
    results_dir_1 = f"model_results/{llm_1.replace('/', '--')}/{type_of_abstract}"
    results_dir_2 = f"model_results/{llm_2.replace('/', '--')}/{type_of_abstract}"
    PPL_A_and_B_1 = np.load(f"{results_dir_1}/PPL_A_and_B.npy")
    PPL_A_and_B_2 = np.load(f"{results_dir_2}/PPL_A_and_B.npy")
    true_labels = np.load(f"{results_dir_1}/labels.npy")
    
    true_label = true_labels[abstract_idx]
    # return True if only llm_1 got correct
    llm_1_pred_label = np.argmin(PPL_A_and_B_1[abstract_idx])
    llm_2_pred_label = np.argmin(PPL_A_and_B_2[abstract_idx])

    if llm_1_pred_label != true_label and llm_2_pred_label == true_label:
        print(f"Abstract {abstract_idx} is not correct for {llm_1}")
        print(f"Abstract {abstract_idx} is correct for {llm_2}")
        return True


def _produce_tokens(llm, abstract):
    """
    Produce tokens for a given abstract using a given tokenizer.
    """
    import transformers
    model_fpath = f"{LOCAL_PATH}/BrainlessGPT/model_training/exp/{llm}/checkpoint.4"
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_fpath)
    token_ids = tokenizer(abstract, return_tensors="pt").input_ids
    # decode token_ids
    tokens = tokenizer.convert_ids_to_tokens(token_ids[0])

    # remove Ġ for readability
    tokens = [token.replace("Ġ", "") for token in tokens]
    tokens = "-".join(tokens)
    return tokens


def _load_data_pairs():
    with open("tokenization_viz_picked.txt", "r") as f:
        pairs = f.read().split("\n\n\n")
        for pair in pairs:
            one = pair.strip().split("\n")[0]
            two = pair.strip().split("\n")[1]
            yield one, two


def _plot_words(ax, words, title, intersection, idx):
    if idx <= 1:
        ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)  # Initial limits, adjusted dynamically later
    ax.set_ylim(0, 6)
    ax.axis('off')  # Hide the axes
    x, y = 1, 5  # Starting coordinates
    max_width = 0

    for word in words:
        # Create text object
        text = ax.text(x, y, word, verticalalignment='center', fontsize=12)
        renderer = ax.figure.canvas.get_renderer()
        bbox = text.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())

        word_width = bbox.width - .5 * (len(word))  # Subtract a small constant to tighten the fit

        # if "neuro" in title.lower():
        #     color = 'lightblue' if word in intersection else 'yellow'
        # else:
        if word in intersection:
            color = 'lightblue'
        else:
            # we randomly sample a color from a list of colors, where the list is 10 colors long
            # to avoid repeating colors
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta']
            color = colors[hash(word) % 10]

        patch = patches.Rectangle((x, y - 0.5), word_width, 1.0, color=color, alpha=0.5, linewidth=0)
        ax.add_patch(patch)

        x += word_width + 0.5  # Increment x by word width plus some padding

        if x > max_width:
            max_width = x  # Update the maximum width to adjust xlim later
        if x + word_width > 98:  # Check if next word exceeds the width limit
            x = 1  # Reset x position
            y -= 1.5  # Move to the next line

    ax.set_xlim(0, max_width + 1)  # Adjust xlim based on the maximum width used


def create_viz_output_txt(llm_1, llm_2):
    abstracts = _load_abstract()
    viz_outputs = open("tokenization_viz.txt", "a")

    for idx, abstract in enumerate(abstracts):
        if not _locate_examples(llm_1, llm_2, idx):
            continue
            
        print(f"Abstract {idx}", file=viz_outputs)
        tokens_1 = _produce_tokens(llm_1, abstract)
        tokens_2 = _produce_tokens(llm_2, abstract)
        print(tokens_1, file=viz_outputs)
        print(tokens_2, file=viz_outputs)
        print("\n\n", file=viz_outputs)
    

def create_viz_output_fig():
    n_pairs = len(list(_load_data_pairs()))
    print(f"Number of pairs: {n_pairs}")
    # Create a figure with two subplots side by side
    n_rows = n_pairs
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    idx = 0

    for row1, row2 in _load_data_pairs():
        print(f"idx: {idx}")
        print(f"row1: {row1}")
        print(f"row2: {row2}")
        dict1 = {word: i for i, word in enumerate(row1.split("-"))}
        dict2 = {word: i for i, word in enumerate(row2.split("-"))}

        # Find the intersection of the two dictionaries
        intersection = set(dict1.keys()) & set(dict2.keys())

        # Plotting the words
        _plot_words(axs[idx], dict1.keys(), 'Pretrain Tokenizer', intersection, idx)
        _plot_words(axs[idx+1], dict2.keys(), 'Neuro Tokenizer', intersection, idx)
        idx += 2

    plt.tight_layout()
    plt.savefig("figs/tokenization_viz.pdf")


def main():
    llm_1 = "gpt2_scratch"
    llm_2 = "gpt2_scratch_neuro_tokenizer"

    if not os.path.exists("tokenization_viz.txt"):
        create_viz_output_txt(llm_1, llm_2)
    
    create_viz_output_fig()


if __name__ == "__main__":
    LOCAL_PATH = env_config("LOCAL_PATH")
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")

    if parser.parse_args().use_human_abstract == "True":
        use_human_abstract = True
    else:
        use_human_abstract = False
    
    if use_human_abstract:
        type_of_abstract = 'human_abstracts'
        abstracts_fpath = "testcases/BrainBench_Human_v0.1.csv"
    else:
        type_of_abstract = 'llm_abstracts'
        abstracts_fpath = "testcases/BrainBench_GPT-4_v0.1.csv"
    
    main()

    
    
