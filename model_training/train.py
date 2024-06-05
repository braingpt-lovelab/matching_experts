import os
import random
import argparse
import math
import pickle
import time
import json
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import SchedulerType, AdamW, get_scheduler
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def tokenize(element, tokenizer, args):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=args.chunk_size,
        return_overflowing_tokens=True,
        return_length=True,
    )
    output_ids = list(itertools.chain(*outputs["input_ids"]))
    output_mask = list(itertools.chain(*outputs["attention_mask"]))
    output_ids = [output_ids[x:x+args.chunk_size] for x in range(0, len(output_ids), args.chunk_size)]
    output_mask = [output_mask[x:x+args.chunk_size] for x in range(0, len(output_mask), args.chunk_size)]
    return {"input_ids": output_ids, "attention_mask": output_mask}


def collate_fn(batch):
    input_ids = [sample["input_ids"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]
    labels = pad_sequence(input_ids, batch_first=True, padding_value=-1)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids,  #.to(device),
        "attention_mask": attention_masks,  #.to(device),
        "labels": labels,  #.to(device),
    }


def save_checkpoint(LLM, tokenizer, outputdir, epoch):
    fulloutput = os.path.join(outputdir, "checkpoint.{}".format(epoch))
    os.system(f"mkdir -p {fulloutput}")
    # save tokenizer
    tokenizer.save_pretrained(fulloutput)
    # save model
    LLM.module.save_pretrained(fulloutput)


def main(rank, args, world_size):
    print(f"rank: {rank}")

    # Save model configuration
    with open(os.path.join(args.outputdir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load huggingface dataset
    dataset = load_dataset(args.data_path, cache_dir=args.cache_dir)

    # Load tokenizer
    if args.custom_tokenizer != "None":
        print(f"Load custom tokenizer from {args.custom_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.custom_tokenizer, cache_dir=args.cache_dir)
    else:
        print(f"Load pretrained tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "args": args},
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    logging("Loading {} samples for training".format(len(tokenized_dataset["train"])), args.logfile)
    tokenized_dataset.set_format("torch")
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    # Define model
    if args.train_mode == "scratch":
        print(f"Train from scratch")
        model_config = AutoConfig.from_pretrained(args.model_path)
        # Update vocab size if using custom tokenizer
        if args.custom_tokenizer != "None":
            model_config.vocab_size = tokenizer.vocab_size
        LLM = AutoModelForCausalLM.from_config(model_config)
    elif args.train_mode == "finetune":
        print(f"Finetune from {args.model_path}")
        LLM = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir=args.cache_dir)
        if args.custom_tokenizer != "None":
            raise ValueError("Bad idea to use custom tokenizer for finetuning?")
    else:
        raise ValueError("Invalid train mode")

    ## Initialise criterion and optimiser
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    ## Optimiser
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in LLM.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in LLM.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = args.num_warmup_steps * max_train_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    LLM, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        LLM, optimizer, train_dataloader, valid_dataloader, lr_scheduler)

    logging("Start training", args.logfile)
    # Training loop
    best_val_loss = 10000
    trainsize = len(train_dataloader)
    for epoch in range(args.num_train_epochs):
        start = time.time()
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            labels = batch["labels"]
            # Maunally remove labels to avoid internal loss compute
            # that does not use ignore_index
            batch = {
                "input_ids": batch["input_ids"], 
                "attention_mask": batch["attention_mask"]
            }
            shift_logits = LLM(**batch).logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )

            loss = loss / args.gradient_accumulation_steps
            # loss.backward()
            accelerator.backward(loss)

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (i + 1) % args.log_interval == 0 and accelerator.is_main_process:
                elasped_time = time.time() - start
                PPL = math.exp(loss.item() * args.gradient_accumulation_steps)
                logging(f"Epoch {epoch} | Batch {i}/{trainsize} | Training PPL: {PPL} | time {elasped_time}", args.logfile)
                accelerator.log({"Epoch": epoch, "Batch": i, "Training PPL": PPL, "Learning Rate": optimizer.param_groups[0]["lr"]})

            if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
                # Evaluate every args.save_interval steps
                LLM.eval()
                with torch.no_grad():
                    val_loss = evaluate(args, LLM, valid_dataloader, criterion)
                    current_lr = optimizer.param_groups[0]["lr"]
                    torch.distributed.reduce(val_loss, 0)
                    val_loss = val_loss / world_size

                    # Save models
                    if accelerator.is_main_process:
                        val_ppl = math.exp(val_loss)
                        logging(f"Epoch {epoch} | Validation PPL: {val_ppl} | Learning rate: {current_lr}", args.logfile)
                        accelerator.log({"Epoch": epoch, "Batch": i, "Validation PPL": val_ppl, "Learning Rate": optimizer.param_groups[0]["lr"]})

                        if val_loss < best_val_loss:
                            ckpt_path = os.path.join(args.outputdir, "checkpoint.{}_{}".format(epoch, (i + 1)))
                            logging(f"Save checkpoint to {ckpt_path}", args.logfile)
                            save_checkpoint(LLM, tokenizer, args.outputdir, f"{epoch}_{(i+1)}")
                LLM.train()
       
        # Evaluate again at the end of epoch
        LLM.eval()
        with torch.no_grad():
            val_loss = evaluate(args, LLM, valid_dataloader, criterion)
            current_lr = optimizer.param_groups[0]["lr"]
            torch.distributed.reduce(val_loss, 0)
            val_loss = val_loss / world_size

            # Save models
            if accelerator.is_main_process:
                val_ppl = math.exp(val_loss)
                logging(f"End of epoch {epoch} | Validation PPL: {val_ppl} | Learning rate: {current_lr}", args.logfile)
                accelerator.log({"End of epoch": epoch, "Validation PPL": val_ppl, "Learning Rate": optimizer.param_groups[0]["lr"]})

                if val_loss < best_val_loss:
                    ckpt_path = os.path.join(args.outputdir, "checkpoint.{}".format(epoch))
                    logging(f"Save checkpoint to {ckpt_path}", args.logfile)
                    save_checkpoint(LLM, tokenizer, args.outputdir, f"{epoch}")
        LLM.train()


def evaluate(args, LLM, valid_dataloader, criterion):
    total_tokens = 0
    total_loss = 0.
    for i, batch in enumerate(valid_dataloader):
        with torch.cuda.amp.autocast():
            labels = batch["labels"]
            batch = {
                "input_ids": batch["input_ids"], 
                "attention_mask": batch["attention_mask"]
            }
            logits = LLM(**batch).logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            ntokens = (batch["attention_mask"][:, 1:] == 1).sum()
            total_tokens += ntokens
            total_loss += loss * ntokens
    return total_loss / total_tokens


if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="BrainlessGPT")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./hf_models",
        help="Path to the model file",
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./cache',
        help='Path to the cache directory'
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./hf_models",
        help="Path to the train data file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="maximum number of tokens in each sample"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default='./exp/clip_vlm',
        help="Path to the output dir",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log interval",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Saving interval",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default='12355',
        help="Master port number",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default='scratch',
        help="Training mode",
        choices=["scratch", "finetune"]
    )
    parser.add_argument(
        "--custom_tokenizer",
        type=str,
        default=None,
        help="Custom tokenizer path",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="brainlessgpt",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="kenotron",
        help="Wandb entity name",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name=args.wandb_project,
        init_kwargs={"wandb": {"entity": args.wandb_entity}},
        config=args.__dict__
    )

    device = accelerator.device
    random.seed(1)
    torch.manual_seed(1)

    main(0, args, world_size)
    accelerator.end_training()