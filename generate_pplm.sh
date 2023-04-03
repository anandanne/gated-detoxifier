cd DExperts
USE_GATE=false


pplm() {
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

    python -m scripts.run_toxicity_experiment \
        --model-type pplm \
        --model $MODEL \
        --prompt $PROMPT \
        --max-tokens $TOKENS \
        --classifier-model $GATE_MODEL \
        --batch-size 2 \
        --n $COUNT \
        --p 0.9 \
        "data-v2/$VERSION/pplm$GATED/$PROMPT-$COUNT($TOKENS).jsonl"
}
pplm "heegyu/gpt2-emotion" "emotion" 10 64
