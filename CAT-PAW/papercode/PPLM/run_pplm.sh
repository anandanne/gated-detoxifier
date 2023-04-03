# python3 run_pplm.py \
#     --pretrained_model heegyu/gpt2-emotion \
#     --cond_text "topic: positive\n" \
#     --discrim gpt2-emotion-toxicity \
#     --top_k 50 \
#     --length 32

python3 pplm.py \
    --cond-text "What the " \
    --discrim toxicity \
    --top_k 50 \
    --print-result \
    --print-intermediate-result \
    --length 32