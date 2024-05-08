import transformers
import torch


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
            "gpt2_scratch_neuro_tokenizer"
        ]:
        model_fpath = f"/home/ken/projects/BrainlessGPT/model_training/exp/{model_fpath}/checkpoint.4"
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