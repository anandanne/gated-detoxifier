VERSION="v3"

sentiment_series() {
    count=$1
    # python evaluate.py "sentiment" "data-v2/$VERSION/gpt2/yelp-$count(32,p1.0).jsonl"
    # python evaluate.py "sentiment" "data-v2/$VERSION/dexpert/yelp-$count(32,A0.5).jsonl"
    # python evaluate.py "sentiment" "data-v2/$VERSION/dexpert/yelp-$count(32,A1.0).jsonl"
    # python evaluate.py "sentiment" "data-v2/$VERSION/dexpert_gated/yelp-$count(32,A0.5).jsonl"
    # python evaluate.py "sentiment" "data-v2/$VERSION/dexpert_gated/yelp-$count(32,A1.0).jsonl"
    
    python evaluate.py "sentiment" "data-v2/$VERSION/gedi/yelp-$count(32,o15,ls10).jsonl"
    python evaluate.py "sentiment" "data-v2/$VERSION/gedi/yelp-$count(32,o30,ls10).jsonl"
    python evaluate.py "sentiment" "data-v2/$VERSION/gedi_gated/yelp-$count(32,o15,ls10).jsonl"
    python evaluate.py "sentiment" "data-v2/$VERSION/gedi_gated/yelp-$count(32,o30,ls10).jsonl"
    # pplm
}

# sentiment_series 500
python evaluate.py "sentiment" "data-v2/$VERSION/pplm_gated/gated_yelp-1000.jsonl"
python evaluate.py "emotion" "data-v2/$VERSION/pplm_gated/gated_emotion-1000.jsonl"
python evaluate.py "bbc-news" "data-v2/$VERSION/pplm_gated/gated_bbc-news-1000.jsonl"

emotion_series() {
    count=$1
    # python evaluate.py "emotion" "data-v2/$VERSION/gpt2/emotion-$count(64).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/dexpert/emotion-$count(64,A0.5).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/dexpert/emotion-$count(64,A1.0).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/dexpert_gated/emotion-$count(64,A0.5).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/dexpert_gated/emotion-$count(64,A1.0).jsonl"
    
    # python evaluate.py "emotion" "data-v2/$VERSION/gedi/emotion-$count(64,o15,ls10).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/gedi/emotion-$count(64,o30,ls10).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/gedi/emotion-$count(64,o2.0,ls1).jsonl"
    python evaluate.py "emotion" "data-v2/$VERSION/gedi_gated/emotion-$count(64,o15,ls10).jsonl"
    python evaluate.py "emotion" "data-v2/$VERSION/gedi_gated/emotion-$count(64,o30,ls10).jsonl"
}
# emotion_series 50

bbc_series() {
    count=$1
    # python evaluate.py "bbc-news" "data-v2/$VERSION/gpt2/bbc-news-$count(128).jsonl"
    # python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert/bbc-news-$count(128,A0.5).jsonl"
    # python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert/bbc-news-$count(128,A1.0).jsonl"
    # python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert_gated/bbc-news-$count(128,A0.5).jsonl"
    # python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert_gated/bbc-news-$count(128,A1.0).jsonl"

    python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert_gated/bbc-news-$count(128,A0.5).jsonl"
    python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert_gated/bbc-news-$count(128,A1.0).jsonl"
}
# bbc_series 500

gedi_series() {
    model=$1
    count=$2
    python evaluate.py "bbc-news" "data-v2/$VERSION/$model/bbc-news-$count(128,o15,ls10).jsonl"
    python evaluate.py "bbc-news" "data-v2/$VERSION/$model/bbc-news-$count(128,o30,ls10).jsonl"
    python evaluate.py "emotion" "data-v2/$VERSION/$model/emotion-$count(64,o15,ls10).jsonl"
    python evaluate.py "emotion" "data-v2/$VERSION/$model/emotion-$count(64,o30,ls10).jsonl"
    # python evaluate.py "sentiment" "data-v2/$VERSION/$model/yelp-$count(32,o15,ls10).jsonl"
    # python evaluate.py "sentiment" "data-v2/$VERSION/$model/yelp-$count(32,o30,ls10).jsonl"
}

# gedi_series "gedi_gated" 500
