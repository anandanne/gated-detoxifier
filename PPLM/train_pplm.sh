mkdir -p model/bbc-news
mkdir -p model/emotion

python run_pplm_discrim_train.py \
    --dataset toxic \
    --pretrained_model "heegyu/gpt2-emotion" \
    --epochs 10 \
    --output_fp model/emotion/ \
    --save_model
    
python run_pplm_discrim_train.py \
    --dataset toxic \
    --pretrained_model "heegyu/gpt2-bbc-news" \
    --epochs 10 \
    --output_fp model/bbc-news/ \
    --save_model