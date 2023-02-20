ALPHA=3.2
EXPERT_SIZE=large
MODEL_DIR=models/experts/sentiment/$EXPERT_SIZE
PROMPTS_DATASET=prompts/sentiment_prompts-10k/neutral_prompts.jsonl
OUTPUT_DIR=generations/sentiment/neutral_prompts/dexperts/${EXPERT_SIZE}_experts/positive/

python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --pos-model $MODEL_DIR/finetuned_gpt2_positive \
    --neg-model $MODEL_DIR/finetuned_gpt2_negative \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $OUTPUT_DIR
