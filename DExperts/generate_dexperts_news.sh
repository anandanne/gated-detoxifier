function run {
    PROMPT=$1
    MODEL=heegyu/gpt2-news-category
    OUTPUT_DIR=test-generations/$PROMPT
    PROMPTS_DATASET=prompts/$PROMPT.jsonl

    python -m scripts.run_toxicity_experiment\
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type gpt2 \
        --model $MODEL \
        --max-tokens 64 \
        --filter_p 0.9 \
        --batch-size 8 \
        --n 1 \
        "$OUTPUT_DIR/gpt2"

    python -m scripts.run_toxicity_experiment\
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model $MODEL \
        --max-tokens 64 \
        --nontoxic-model heegyu/gpt2-non-toxic \
        --toxic-model heegyu/gpt2-toxic \
        --filter_p 0.9 \
        --alpha 0.5 \
        --batch-size 2 \
        --n 1 \
        "$OUTPUT_DIR/dexperts"

}

# run "news-25"
run "news-all"