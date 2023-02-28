function run {
    PROMPT=$1
    VERSION=$2
    MODEL=heegyu/gpt2-sentiment
    OUTPUT_DIR=test-generations/$PROMPT-$VERSION
    PROMPTS_DATASET=prompts/$PROMPT.jsonl

    python -m scripts.run_toxicity_experiment\
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type gpt2 \
        --model $MODEL \
        --max-tokens 32 \
        --filter_p 0.9 \
        --batch-size 8 \
        --n 1 \
        "$OUTPUT_DIR/gpt2"

    python -m scripts.run_toxicity_experiment\
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model $MODEL \
        --max-tokens 32 \
        --nontoxic-model heegyu/gpt2-non-toxic \
        --toxic-model heegyu/gpt2-toxic \
        --filter_p 0.9 \
        --alpha 0.5 \
        --batch-size 2 \
        --n 1 \
        "$OUTPUT_DIR/dexperts"

}

run "sentiment-20" "v2"
# run "news-all"