expdir=exp/gpt2
mkdir -p $expdir

# python 
accelerate launch --config_file accel_config.yaml finetune.py \
    --model_path gpt2 \
    --data_path BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset_mini \
    --cache_dir cache \
    --batch_size 1 \
    --chunk_size 512 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --num_warmup_steps 0.03 \
    --weight_decay 0.001 \
    --lr_scheduler_type cosine \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 1000 \
    --save_interval 10000 \