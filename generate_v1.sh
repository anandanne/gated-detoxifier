
cd DExperts

VERSION="v4"

gpt2() {
    MODEL=$1
    PROMPT=$2
    TOKENS=$3
    PROMPTS_DATASET=prompts/$PROMPT.jsonl

    python -m scripts.run_toxicity_experiment\
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type gpt2 \
        --model $MODEL \
        --max-tokens $TOKENS \
        --batch-size 1 \
        --n 1 \
        "../data/$VERSION/dexperts-gpt2/$PROMPT"
}

gated_dexperts() {
    MODEL=$1
    PROMPT=$2
    TOKENS=$3
    PROMPTS_DATASET=prompts/$PROMPT.jsonl

    python -m scripts.run_toxicity_experiment\
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model $MODEL \
        --max-tokens $TOKENS \
        --nontoxic-model heegyu/gpt2-non-toxic \
        --toxic-model heegyu/gpt2-toxic \
        --classifier-model s-nlp/roberta_toxicity_classifier \
        --filter_p 0.9 \
        --alpha $ALPHA \
        --batch-size 1 \
        --n 1 \
        "../data/$VERSION/gated-dexperts/$PROMPT-A$ALPHA/"
}

dexperts() {
    MODEL=$1
    PROMPT=$2
    TOKENS=$3
    PROMPTS_DATASET=prompts/$PROMPT.jsonl

    python -m scripts.run_toxicity_experiment\
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model $MODEL \
        --max-tokens $TOKENS \
        --nontoxic-model heegyu/gpt2-non-toxic \
        --toxic-model heegyu/gpt2-toxic \
        --filter_p 0.9 \
        --alpha $ALPHA \
        --batch-size 1 \
        --n 1 \
        "../data/$VERSION/dexperts/$PROMPT-A$ALPHA/"
}

series() {
    model=$1
    prompt=$2
    len=$3

    gpt2 $model "$prompt-$count" $3
    
    ALPHA=0.5
    dexperts $model "$prompt-$count" $3
    gated_dexperts $model "$prompt-$count" $3
    
    ALPHA=1.0
    dexperts $model "$prompt-$count" $3
    gated_dexperts $model "$prompt-$count" $3
}

count=1000
# series "heegyu/gpt2-yelp-polarity" "sentiment" 32
# series "heegyu/gpt2-emotion" "emotion" 32
# series "heegyu/gpt2-news-category" "news" 48
# gpt2 "heegyu/gpt2-emotion" "emotion-10" 64
gpt2 "heegyu/gpt2-bbc-news" "bbc-news-10" 400

function gedi {
    MODEL=$1
    PROMPT=$2
    TOKENS=$3
    PROMPTS_DATASET=prompts/$PROMPT.jsonl

    python -m scripts.run_toxicity_experiment\
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type gedi \
        --model $MODEL \
        --max-tokens $TOKENS \
        --nontoxic-model heegyu/gpt2-non-toxic \
        --toxic-model heegyu/gpt2-toxic \
        --alpha $ALPHA \
        --batch-size 2 \
        --n 1 \
        "../data/$VERSION/gedi/$PROMPT-A$ALPHA/"

        # --filter_p 0.8 \
}

# ALPHA=0.5
# gedi "heegyu/gpt2-yelp-polarity" "sentiment-$COUNT" 32
# gedi "heegyu/gpt2-emotion" "emotion-$COUNT" 32
# gedi "heegyu/gpt2-news-category" "news-$COUNT" 48

# ALPHA=1.0
# gedi "heegyu/gpt2-yelp-polarity" "sentiment-$COUNT" 32
# gedi "heegyu/gpt2-emotion" "emotion-$COUNT" 32
# gedi "heegyu/gpt2-news-category" "news-$COUNT" 48

# gedi "heegyu/gpt2-yelp-polarity" "sentiment-1000" 32
# gedi "heegyu/gpt2-emotion" "emotion-1000" 32
# gedi "heegyu/gpt2-news-category" "news-1000" 48