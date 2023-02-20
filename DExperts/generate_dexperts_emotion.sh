PROMPT=emotion-5
MODEL=heegyu/gpt2-emotion
OUTPUT_DIR=test-generations/$PROMPT
PROMPTS_DATASET=prompts/$PROMPT.jsonl

python -m scripts.run_toxicity_experiment\
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt2 \
    --model $MODEL \
    --filter_p 0.9 \
    "$OUTPUT_DIR/gpt2"

python -m scripts.run_toxicity_experiment\
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model $MODEL \
    --nontoxic-model heegyu/gpt2-non-toxic \
    --toxic-model heegyu/gpt2-toxic \
    --filter_p 0.9 \
    --alpha 0.5 \
    "$OUTPUT_DIR/dexperts"
