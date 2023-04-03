

python3 run_pplm.py \
    --pretrained_model heegyu/gpt2-emotion \
    --cond_text "topic: positive\n" \
    --discrim gpt2-emotion-toxicity \
    --top_k 50 \
    --length 32