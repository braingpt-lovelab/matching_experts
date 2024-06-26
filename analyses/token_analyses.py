import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib.gridspec import GridSpec


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

    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

    # Create figure and define grid layout
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, figure=fig)

    # Top plot - Venn Diagram
    ax1 = fig.add_subplot(gs[0, :])  # Span both columns at the top
    total = 1.0
    shared = common_proportion * total
    exclusive = (total - shared) / 2

    venn_diagram = venn2(
        subsets=(exclusive, exclusive, shared), 
        # set_labels=('Pretrain\nVocab.', 'Neuro Tokenizer\nVocab.'),
        set_labels=('', ''),
        ax=ax1
    )
    
    venn_diagram.get_patch_by_id('10').set_alpha(1)
    venn_diagram.get_patch_by_id('01').set_alpha(1)
    venn_diagram.get_patch_by_id('11').set_alpha(0.5)
    venn_diagram.get_patch_by_id('10').set_color('skyblue')
    venn_diagram.get_patch_by_id('01').set_color('lightgreen')
    venn_diagram.get_patch_by_id('11').set_color('grey')
    venn_diagram.get_label_by_id('10').set_text('')
    venn_diagram.get_label_by_id('01').set_text('')
    venn_diagram.get_label_by_id('11').set_text(f'Shared\nTokens\n{shared*100:.1f}%')

    # Add label A to upper left corner of ax1
    ax1.text(-0.8, 0.5, 'A', fontsize=16, fontweight='bold')

    # Bottom left - Pretrain Vocab Pie Chart
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.pie([pretrain_proportion, 1 - pretrain_proportion], labels=["Neuro\nTokens", ""],
            autopct='%1.1f%%', startangle=90, explode=(0.1, 0.), colors=['cyan', 'skyblue'])
    ax2.set_xlabel("Pretrain\nVocab.", fontsize=16, fontweight='bold')
    ax2.text(-2, 1.5, 'B', fontsize=16, fontweight='bold')

    # Bottom right - Neuro Tokenizer Vocab Pie Chart
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.pie([neuro_proportion, 1 - neuro_proportion], labels=["Neuro\nTokens", ""],
            autopct='%1.1f%%', startangle=90, explode=(0.1, 0.), colors=['cyan', 'lightgreen'])
    ax3.set_xlabel("Neuro Tokenizer\nVocab.", fontsize=16, fontweight='bold')
    ax3.text(-2, 1.5, 'C', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figs/shared_terms_and_neuro_term_proportion.pdf")
    

if __name__ == "__main__":
    results_dir = "token_results"
    pretrain_vocab = json.load(open('data/vocab_gpt2_pretrain.json'))
    neuro_vocab = json.load(open('data/vocab_gpt2_neuro.json'))

    pretrain_neuro_term_fname = "pretrain_filtered__neuro_term_tagging_results.txt"
    neuro_neuro_term_fname = "neuro_filtered__neuro_term_tagging_results.txt"

    main()