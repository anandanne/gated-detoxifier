
cd DExperts

VERSION="v2"
COUNT=1000

ALPHA=0.5
function dexperts {
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
        --batch-size 2 \
        --n 1 \
        "../data/$VERSION/dexperts/$PROMPT-A$ALPHA/"

}

ALPHA=0.5
dexperts "heegyu/gpt2-yelp-polarity" "sentiment-$COUNT" 32
dexperts "heegyu/gpt2-emotion" "emotion-$COUNT" 32
dexperts "heegyu/gpt2-news-category" "news-$COUNT" 48

ALPHA=1.0
dexperts "heegyu/gpt2-yelp-polarity" "sentiment-$COUNT" 32
dexperts "heegyu/gpt2-emotion" "emotion-$COUNT" 32
dexperts "heegyu/gpt2-news-category" "news-$COUNT" 48

# dexperts "heegyu/gpt2-yelp-polarity" "sentiment-1000" 32
# dexperts "heegyu/gpt2-emotion" "emotion-1000" 32
# dexperts "heegyu/gpt2-news-category" "news-1000" 48

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

ALPHA=0.5
gedi "heegyu/gpt2-yelp-polarity" "sentiment-$COUNT" 32
gedi "heegyu/gpt2-emotion" "emotion-$COUNT" 32
gedi "heegyu/gpt2-news-category" "news-$COUNT" 48

ALPHA=1.0
gedi "heegyu/gpt2-yelp-polarity" "sentiment-$COUNT" 32
gedi "heegyu/gpt2-emotion" "emotion-$COUNT" 32
gedi "heegyu/gpt2-news-category" "news-$COUNT" 48

# gedi "heegyu/gpt2-yelp-polarity" "sentiment-1000" 32
# gedi "heegyu/gpt2-emotion" "emotion-1000" 32
# gedi "heegyu/gpt2-news-category" "news-1000" 48