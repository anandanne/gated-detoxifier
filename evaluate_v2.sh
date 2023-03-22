VERSION="v3"

emotion_series() {
    count=$1
    python evaluate.py "emotion" "data-v2/$VERSION/gpt2/emotion-$count(64).jsonl"
    python evaluate.py "emotion" "data-v2/$VERSION/dexpert/emotion-$count(64,A0.5).jsonl"
    python evaluate.py "emotion" "data-v2/$VERSION/dexpert/emotion-$count(64,A1.0).jsonl"
    python evaluate.py "emotion" "data-v2/$VERSION/dexpert_gated/emotion-$count(64,A0.5).jsonl"
    python evaluate.py "emotion" "data-v2/$VERSION/dexpert_gated/emotion-$count(64,A1.0).jsonl"
    
    # python evaluate.py "emotion" "data-v2/$VERSION/dexpert/emotion-$count(64,A1.0).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/gedi/emotion-$count(64,o0.5,ls1).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/gedi/emotion-$count(64,o1,ls1).jsonl"
    # python evaluate.py "emotion" "data-v2/$VERSION/gedi/emotion-$count(64,o2.0,ls1).jsonl"
}
emotion_series 500

bbc_series() {
    count=$1
    python evaluate.py "bbc-news" "data-v2/$VERSION/gpt2/bbc-news-$count(128).jsonl"
    python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert/bbc-news-$count(128,A0.5).jsonl"
    python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert/bbc-news-$count(128,A1.0).jsonl"
    python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert_gated/bbc-news-$count(128,A0.5).jsonl"
    python evaluate.py "bbc-news" "data-v2/$VERSION/dexpert_gated/bbc-news-$count(128,A1.0).jsonl"
}
bbc_series 500

# gedi_series() {
#     count=$1
#     # python evaluate.py "sentiment" "data-v2/$VERSION/$model/sentiment-10(64).jsonl"
#     python evaluate.py "bbc-news" "data-v2/$VERSION/gedi/bbc-news-$count(400,o15,ls10).jsonl"
# }

# # gedi_series 100

python evaluate.py "emotion" "data-v2/v3/dexpert/emotion-1000(64,A0.5).jsonl"