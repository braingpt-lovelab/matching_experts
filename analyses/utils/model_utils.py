import transformers
import torch


model_list = {
    "gpt2": {
        "gpt2_init": {
            "llm": "GPT2-Untrained (124M)",
            "color": '#A5CAD2',
            "alpha": 0.3,
            "hatch": "",
        },
        "gpt2": {
            "llm": "GPT2-Pretrained (124M)",
            "color": '#758EB7',
            "alpha": 0.4,
            "hatch": "",
        },
        "gpt2_scratch": {
            "llm": "GPT2-Scratch (124M)",
            "color": '#E1C0D8',
            "alpha": 0.6,
            "hatch": "",
        },
        "finetune_gpt2": {
            "llm": "GPT2-Finetuned (124M)",
            "color": '#D2A9B0',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2_scratch_neuro_tokenizer": {
            "llm": "GPT2-Neuro (124M)",
            "color": '#FA9284',
            "alpha": 0.9,
            "hatch": "",
        },
        "gpt2-medium_scratch_neuro_tokenizer": {
            "llm": "GPT2-Neuro (355M)",
            "color": '#FA9284',
            "alpha": 0.9,
            "hatch": "",
        },
        "gpt2-large_scratch_neuro_tokenizer": {
            "llm": "GPT2-Neuro (774M)",
            "color": '#FA9284',
            "alpha": 0.9,
            "hatch": "",
        },
    },
}


def load_model_and_tokenizer(model_fpath, tokenizer_only=False):
    if tokenizer_only:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_fpath,
        )
        return tokenizer
    
    load_in_8bit = False
    torch_dtype = torch.float16


    # Load model trained from scratch from local checkpoint
    if model_fpath in [
            "gpt2_scratch",
            "finetune_gpt2",
            "finetune_gpt2_lr2e-6",
            "gpt2_scratch_neuro_tokenizer",
            "gpt2_scratch_neuro_tokenizer_spt",
            "gpt2-medium_scratch_neuro_tokenizer",
            "gpt2-medium_scratch_neuro_tokenizer_spt",
            "gpt2-large_scratch_neuro_tokenizer",
            "gpt2-large_scratch_neuro_tokenizer_spt",
        ]:
        model_fpath = f"/home/ken/projects/matching_experts/model_training/exp/{model_fpath}/checkpoint.4"
        print("Loading GPT2 model from", model_fpath)
        model = transformers.GPT2LMHeadModel.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            model_fpath,    
        )
    
    # Load model untrained (config only)
    elif model_fpath == "gpt2_init":
        print("Loading GPT2 model untrained")
        from transformers import AutoConfig, AutoModelForCausalLM
        model_config = AutoConfig.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_config(model_config).to('cuda')
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            "gpt2",
        )

    # Load pretrained model
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_fpath,
        )

    return model, tokenizer