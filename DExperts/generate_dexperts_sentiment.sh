OUTPUT_DIR=test-generations/toxicity/emotion
PROMPTS_DATASET=prompts/emotion-10.jsonl
MODEL=heegyu/gpt2-emotion

python -m scripts.run_toxicity_experiment\
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model $MODEL \
    --nontoxic-model heegyu/gpt2-non-toxic \
    --toxic-model heegyu/gpt2-toxic \
    --filter_p 0.9 \
    --alpha 0.5 \
    $OUTPUT_DIR
