import json
from datasets import load_dataset

# Load json file `configs/gpt2_test.json`
args = json.load(open('configs/gpt2_test.json'))
dataset = load_dataset(args["data_path"], cache_dir=args["cache_dir"])

def get_training_corpus(dataset):
    return (
        dataset["train"][i : i + 1000]["text"]
        for i in range(0, len(dataset["train"]), 1000)
    )

training_corpus = get_training_corpus(dataset)

from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
example = next(training_corpus)[:1][:5][0]
print(example)
print(old_tokenizer.tokenize(example))

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, old_tokenizer.vocab_size)
print(tokenizer.tokenize(example))
print(tokenizer.vocab_size)

# Save the new tokenizer
tokenizer.save_pretrained("gpt2_neuro_tokenizer")