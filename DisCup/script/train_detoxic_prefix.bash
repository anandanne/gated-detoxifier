gpu=0
temperature=1.0
batch_size=150
file_name='eval'
out_dir='../detoxic_model/prefix/gpt2-emotion'
model_name_or_path="heegyu/gpt2-emotion"
# out_dir='../detoxic_model/prefix'
# model_name_or_path="/home/zhanghanqing/pretrained_model/gpt2/large"
mode="train"
data_path="../datasets/toxic/positive.txt"
task_name="detoxic"
template="(5,5)"
corpus_type="positive"
tuning_name="prompt_tuning"
epoch=15


for lr_discrimator in 1.0
do
    echo  ---epoch--$epoch---------
    CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size  --file_name $file_name --out_dir $out_dir --epoch $epoch --model_name_or_path $model_name_or_path --mode $mode --data_path=$data_path --template $template --corpus_type $corpus_type  --tuning_name $tuning_name --task_name $task_name

    wait
done