ids=$1

WORLD_SIZE=$GPU_PER_NODE torchrun --nproc_per_node $GPU_PER_NODE \
    main.py \
    --model_name 'facebook/blenderbot-400M-distill' \
    --wandb_run_name 'blenderbot-RMT' \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --weight_decay 0 \
    --eval_steps 1000 \
    --max_steps 100000 \
    --report_to 'wandb' \
    --output_dir 'outputs' \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --save_steps 1000 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing True

    # --use_lora --fp16 --train_8bit
