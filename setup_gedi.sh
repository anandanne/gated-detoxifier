
# git clone https://github.com/salesforce/GeDi.git
cd GeDi

# '''Installing transformers v2.8'''
# pip install transformers==2.8 datasets jsonlines
# pip install -r hf_requirements.txt

# '''Downloading GeDi topic model checkpoints'''
# wget https://storage.googleapis.com/sfr-gedi-data/gedi_detoxifier.zip
# unzip gedi_detoxifier.zip

# git clone https://huggingface.co/heegyu/gpt2-yelp-polarity
# cd gpt2-yelp-polarity
# git lfs pull


git clone https://huggingface.co/heegyu/gpt2-emotion
cd gpt2-emotion
git lfs pull

git clone https://huggingface.co/heegyu/gpt2-news-category
cd gpt2-news-category
git lfs pull