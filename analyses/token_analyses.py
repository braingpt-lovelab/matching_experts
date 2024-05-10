import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2


def _load_tagging_results(fname):
    """
    Load and convert to a dictionary,
    each entry is a row, key is the word, value is the result.
    """
    tagging_map = {}
    with open(f"{results_dir}/{fname}", 'r') as f:
        lines = f.readlines()
        for line in lines:
            word, result = line.strip().split(", ")
            if result == "yes":
                result = 1
            else:
                result = 0
            tagging_map[word] = result
    return tagging_map


def _get_proportion(tagging_map, vocab):
    n_neuro_terms = 0
    for token in vocab:
        token = token.replace('\u0120', '').lower()
        if token in tagging_map and tagging_map[token] == 1:
            n_neuro_terms += 1
    return n_neuro_terms / len(vocab)


def common_term_proportion():
    """
    Check shared tokens in both pretrain and neuro vocabs.
    """
    common_terms = set(pretrain_vocab.keys()).intersection(neuro_vocab.keys())
    print(f"common term proportion: {len(common_terms) / len(pretrain_vocab):.2f}")
    return len(common_terms) / len(pretrain_vocab)


def neuro_term_proportion():
    """
    Compare proportion of tokens (deemed by GPT) to be a common
    term in Neuroscience.
    """
    pretrain_tagging_map = _load_tagging_results(pretrain_neuro_term_fname)
    neuro_tagging_map = _load_tagging_results(neuro_neuro_term_fname)

    pretrain_proportion = _get_proportion(pretrain_tagging_map, pretrain_vocab)
    neuro_proportion = _get_proportion(neuro_tagging_map, neuro_vocab)

    print(f"pretrain neuro term proportion: {pretrain_proportion:.2f}")
    print(f"neuro tokenizer neuro term proportion: {neuro_proportion:.2f}")
    return pretrain_proportion, neuro_proportion


def main():
    common_proportion = common_term_proportion()
    pretrain_proportion, neuro_proportion = neuro_term_proportion()

    # Left plots 2 intersected pies the intersection has the common_proportion
    # Right plots 2 individual pies, each shows the proportion of neuro terms in each vocab.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    
    # Venn diagram for Shared and Unique
    # The subsets parameter needs proper values for exclusive and shared areas.
    total = 1.0  # Normalize total to 1 for simplicity in this example
    shared = common_proportion * total
    exclusive = (total - shared) / 2  # Assuming equal exclusive parts for simplicity

    venn_diagram = venn2(
        subsets=(exclusive, exclusive, shared), 
        set_labels=('Pretrain\nVocab.', 'Neuro Tokenizer\nVocab.'),
        ax=axes[0]
    )

    venn_diagram.get_patch_by_id('10').set_color('skyblue')
    venn_diagram.get_patch_by_id('01').set_color('lightgreen')
    venn_diagram.get_patch_by_id('11').set_color('salmon')

    venn_diagram.get_label_by_id('10').set_text('')
    venn_diagram.get_label_by_id('01').set_text('')
    venn_diagram.get_label_by_id('11').set_text(f'Shared Tokens\n{shared*100:.2f}%')
    
    axes[1].pie([pretrain_proportion, 1 - pretrain_proportion], labels=["Neuro Tokens", ""],
                autopct='%1.2f%%', startangle=90, explode=(0.1, 0.), colors=['cyan', 'skyblue'])
    axes[1].set_xlabel("Pretrain\nVocab.", fontsize=14, fontweight='bold')

    axes[2].pie([neuro_proportion, 1 - neuro_proportion], labels=["Neuro Tokens", ""],
                autopct='%1.2f%%', startangle=90, explode=(0.1, 0.), colors=['cyan', 'lightgreen'])
    axes[2].set_xlabel("Neuro Tokenizer\nVocab.", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figs/shared_terms_and_neuro_term_proportion.pdf")
    

if __name__ == "__main__":
    results_dir = "token_results"
    pretrain_vocab = json.load(open('data/vocab_gpt2_pretrain.json'))
    neuro_vocab = json.load(open('data/vocab_gpt2_neuro.json'))

    pretrain_neuro_term_fname = "pretrain_filtered__neuro_term_tagging_results.txt"
    neuro_neuro_term_fname = "neuro_filtered__neuro_term_tagging_results.txt"

    main()