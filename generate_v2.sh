VERSION="v1"


gpt2() {
    # gpt2 $model $prompt $count $tokens
    MODEL=$1
    PROMPT=$2
    COUNT=$3
    TOKENS=$4

    python run_generation.py \
        --model-type gpt2 \
        --model $MODEL \
        --prompt $PROMPT \
        --max-tokens $TOKENS \
        --batch-size 2 \
        --n $COUNT \
        --p 0.9 \
        "data-v2/$VERSION/gpt2/$PROMPT-$COUNT($TOKENS,p0.9).jsonl"
}


# gpt2 "heegyu/gpt2-emotion" "emotion" 10 64
# gpt2 "heegyu/gpt2-bbc-news" "bbc-news" 10 400
USE_GATE=false
ALPHA=1.0
dexpert() {
    # dexpert $model $prompt $count $tokens
    MODEL=$1
    PROMPT=$2
    COUNT=$3
    TOKENS=$4
    if [ $USE_GATE = true ] 
    then 
        GATE_MODEL="s-nlp/roberta_toxicity_classifier"
        GATED="_gated"
    else
        GATE_MODEL="no"
        GATED=""
    fi

    python run_generation.py \
        --model-type dexperts \
        --model $MODEL \
        --prompt $PROMPT \
        --max-tokens $TOKENS \
        --nontoxic-model heegyu/gpt2-non-toxic \
        --toxic-model heegyu/gpt2-toxic \
        --classifier-model $GATE_MODEL \
        --batch-size 2 \
        --n $COUNT \
        --alpha $ALPHA \
        --overwrite \
        --p 0.9 \
        "data-v2/$VERSION/dexpert$GATED/$PROMPT-$COUNT($TOKENS,A$ALPHA).jsonl"
}
# dexpert "heegyu/gpt2-emotion" "emotion" 10 64
# dexpert "heegyu/gpt2-bbc-news" "bbc-news" 10 400


OMEGA=30
LOGITS_SCALE=10
gedi() {
    # dexpert $model $prompt $count $tokens
    MODEL=$1
    PROMPT=$2
    COUNT=$3
    TOKENS=$4

    if [ $USE_GATE = true ] 
    then
        GATE_MODEL="s-nlp/roberta_toxicity_classifier"
        GATED="_gated"
    else
        GATE_MODEL="no"
        GATED=""
    fi
    cd GeDi 

    python run_gedi.py \
        --model $MODEL \
        --prompt $PROMPT \
        --max-tokens $TOKENS \
        --classifier-model $GATE_MODEL \
        --batch-size 2 \
        --n $COUNT \
        --disc_weight $OMEGA \
        --logits_scale $LOGITS_SCALE \
        --overwrite \
        --top_p 0.9 \
        "../data-v2/$VERSION/gedi$GATED/$PROMPT-$COUNT($TOKENS,o$OMEGA,ls$LOGITS_SCALE).jsonl"

    cd ..
}

# gedi "heegyu/gpt2-emotion" "emotion" 100 64
# gedi "heegyu/gpt2-bbc-news" "bbc-news" 1 128

series() {
    MODEL=$1
    PROMPT=$2
    COUNT=$3
    TOKENS=$4
    
    # gpt2 $MODEL $PROMPT $COUNT $TOKENS

    # USE_GATE=false
    # ALPHA=0.5
    # dexpert $MODEL $PROMPT $COUNT $TOKENS
    # ALPHA=1.0
    # dexpert $MODEL $PROMPT $COUNT $TOKENS

    # USE_GATE=true
    # ALPHA=0.5
    # dexpert $MODEL $PROMPT $COUNT $TOKENS
    # ALPHA=1.0
    # dexpert $MODEL $PROMPT $COUNT $TOKENS

    
    LOGITS_SCALE=10
    USE_GATE=true
    OMEGA=15
    gedi $MODEL $PROMPT $COUNT $TOKENS
    OMEGA=30
    gedi $MODEL $PROMPT $COUNT $TOKENS
}

# series "heegyu/gpt2-yelp-polarity" "yelp" 500 32
# series "heegyu/gpt2-bbc-news" "bbc-news" 500 128
# series "heegyu/gpt2-emotion" "emotion" 500 64
