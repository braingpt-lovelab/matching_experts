import os
import argparse
import numpy as np
import pandas as pd
import transformers
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


# def _load_brainbench_results(llm):
#     results_dir = f"model_results/{llm.replace('/', '--')}/{type_of_abstract}"
#     PPL_A_and_B = np.load(f"{results_dir}/PPL_A_and_B.npy")
#     true_labels = np.load(f"{results_dir}/labels.npy")
#     return PPL_A_and_B, true_labels

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
    model_fpath = f"{LOCAL_PATH}/BrainlessGPT/model_training/exp/{llm}/checkpoint.4"
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_fpath)
    token_ids = tokenizer(abstract, return_tensors="pt").input_ids
    # decode token_ids
    tokens = tokenizer.convert_ids_to_tokens(token_ids[0])

    # remove Ġ for readability
    tokens = [token.replace("Ġ", "") for token in tokens]
    tokens = "-".join(tokens)
    return tokens


def main():
    llm_1 = "gpt2_scratch"
    llm_2 = "gpt2_scratch_neuro_tokenizer"
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

    
    
