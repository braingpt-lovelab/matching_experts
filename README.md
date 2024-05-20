# Matching domain experts by training from scratch on domain knowledge

### To work with the repo locally:
```
git clone git@github.com:braingpt-lovelab/matching_experts.git --recursive
```

### Structure
Training related scripts and files are in `model_training/`; post-training analyses scripts and results are in `analyses/`.

### Training
`cd model_training`
1. To build the neuro-tokenizer: `python tokenizer.py`
2. To train a GPT-2 using a specific configuration: `bash launch_training.sh`. Configurations are in `configs/`
3. Make sure to supply wandb info in the config json.
4. Accelerate config: `accel_config.yaml`

### Reproduce analyses from scratch:
`cd analyses`
1. Run inference with GPT-2 variants on BrainBench test cases: `python run_choice.py`
2. Produce token analysis intermediate results: `python common_and_unique.py`
3. Call GPT-4 to identify neuroscience terms in GPT-2 pretrained tokenizer and neuro-tokenizer vocab: `python neuro_term_tagging.py`

### Plot figures in the paper
`cd analyses`
* Fig 1: `python model_vs_human.py`
* Fig 2: `python token_analyses.py`
* Fig 3: `python tokenization_viz.py`

### Attribution
```
@misc{luo2024matching,
      title={Matching domain experts by training from scratch on domain knowledge}, 
      author={Xiaoliang Luo and Guangzhi Sun and Bradley C. Love},
      year={2024},
      eprint={2405.09395},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```
