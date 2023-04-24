
run() {
    task=$1
    length=$2
    count=$3

    python3 pplm_gated.py \
        --prompt $task \
        --model $task \
        --top_k 50 \
        --top_p 0.9 \
        --print-result \
        --length $length \
        --n $count \
        --label-class 1 \
        --sample \
        --output-file "gated_$task-$count.jsonl"
}


run yelp 32 1000
run emotion 64 1000
run bbc-news 128 1000