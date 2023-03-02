


# conda create -n gedi python=3.8
cd GeDi

function run {
    MODEL=$1
    PROMPT=$2
    TOKENS=$3
    PROMPTS_DATASET=../DExperts/prompts/$PROMPT.jsonl

    python run_gedi.py \
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model $MODEL \
        --n 1 \
        --max-tokens $TOKENS \
        "../data/v1/gedi-v6/$PROMPT/"
}

# run "gpt2-yelp-polarity" "sentiment-10" 32
run "gpt2-emotion" "emotion-10" 32
# run "gpt2-news-category" "news-10" 48

# run "heegyu/gpt2-yelp-polarity" "sentiment-1000" 32
# run "heegyu/gpt2-emotion" "emotion-1000" 32
# run "heegyu/gpt2-news-category" "news-1000" 48