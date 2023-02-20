
# PROMPTS_DATASET=prompts/toy_emotion.jsonl
# MODEL=heegyu/gpt2-emotion

OUTPUT_DIR=test-generations/toxicity/news1
PROMPTS_DATASET=prompts/news.jsonl
MODEL=heegyu/gpt2-news-category

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model $MODEL \
    --nontoxic-model heegyu/gpt2-non-toxic \
    --toxic-model heegyu/gpt2-toxic \
    --filter_p 0.9 \
    --alpha 1.0 \
    $OUTPUT_DIR
