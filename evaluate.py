from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import jsonlines
from tqdm.auto import tqdm
from collections import defaultdict
import click
import re
import pandas as pd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def clean_text(x):
    y = re.sub(r"[^a-zA-Z\"'\!\?\s\.\,\&\(\)\-]", "", x)
    # print(x,"->",y)
    return y

class Classifier:

    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def classify_items(self, items, batch_size: int, id2label, clean=False):
        predictions = []

        for i in tqdm(range(0, len(items), batch_size)):
            end = min(i + batch_size, len(items))
            batch = items[i:end]
            batch = [x['generation'] for x in batch]
            if clean:
                batch = [clean_text(x) for x in batch]

            tokens = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            logits = self.model(**tokens).logits
            preds = logits.argmax(-1).cpu().tolist()
            # print(logits, preds)
            preds = [id2label[p] for p in preds]
            predictions.extend(preds)
        
        return predictions

def score_accuracy(items, preds, use_prompt_label=True, true_label=None):
    total = defaultdict(lambda: 0)
    correct = defaultdict(lambda: 0)

    for item, p in zip(items, preds):
        label = item['prompt'].split(" ")[0]
        total[label] += 1
        # print(label, p)
        if use_prompt_label:
            if label == p:
                correct[label] += 1
        elif p == true_label:
            correct[label] += 1
    # print(items)
    # print(preds)
    # print(list(correct.keys()), list(total.keys()))
    for k in total.keys():
        # print(k, correct[k] / total[k] ,f"{correct[k]} of {total[k]}")
        correct[k] = correct.get(k, 0) / total[k]

    return correct


def eval_news_category(filename, items):
    classifier = Classifier("heegyu/roberta-base-news-category-top10")
    id2label = [
        "ENTERTAINMENT", "POLITICS", "WELLNESS", "TRAVEL", "STYLE",
        "PARENTING", "HEALTHY", "QUEER", "FOOD", "BUSINESS"
        ]
    predictions = classifier.classify_items(items, 8, id2label)

    # print("Topic score for", filename)
    return score_accuracy(items, predictions)

def eval_bbc_news(filename, items):
    # classifier = Classifier("abhishek/autonlp-bbc-news-classification-37229289")
    classifier = Classifier("Umesh/distilbert-bbc-news-classification")
    id2label = [
        "business", "entertainment", "politics", "sport", "tech", 
    ]
    predictions = classifier.classify_items(items, 8, id2label)

    for item in items:
        item["prompt"] = item["prompt"].replace("topic:", "").strip()

    # print("Topic score for", filename)
    return score_accuracy(items, predictions)


def eval_emotion(filename, items):
    # classifier = Classifier("j-hartmann/emotion-english-distilroberta-base")
    # id2label = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    # classifier = Classifier("transformersbook/distilbert-base-uncased-finetuned-emotion")
    # id2label = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    # classifier = Classifier("Aron/distilbert-base-uncased-finetuned-emotion")
    # id2label = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    classifier = Classifier("bhadresh-savani/bert-base-uncased-emotion")
    id2label = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    for item in items:
        item["prompt"] = item["prompt"].replace("topic:", "").strip()

    predictions = classifier.classify_items(items, 8, id2label, True)

    # print("Emotion score for", filename)
    return score_accuracy(items, predictions)

def eval_sentiment(filename, items):
    # yelp: VictorSanh/roberta-base-finetuned-yelp-polarity
    # textattack/bert-base-uncased-yelp-polarity
    # imdb: wrmurray/roberta-base-finetuned-imdb
    classifier = Classifier("VictorSanh/roberta-base-finetuned-yelp-polarity")
    id2label = ["negative", "positive"]
    for item in items:
        item["prompt"] = item["prompt"].replace("topic:", "").strip()
        # item["generation"] = item["generation"].lower()
    # classifier = Classifier("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    # id2label = ["negative", "neutral", "positive"]
    predictions = classifier.classify_items(items, 8, id2label, True)

    # print("Sentiment score for", filename)
    return score_accuracy(items, predictions)

def eval_toxicity(filename, items):
    classifier = Classifier("s-nlp/roberta_toxicity_classifier")
    id2label = ["non-toxic", "toxic"]
    predictions = classifier.classify_items(items, 8, id2label, True)

    # print("Toxicity score for", filename)
    return score_accuracy(items, predictions, False, true_label="toxic")

def eval_grammar(filename, items):
    classifier = Classifier("cointegrated/roberta-large-cola-krishna2020")
    id2label = ["correct", "error"]
    predictions = classifier.classify_items(items, 8, id2label, True)

    # print("Grammar score for", filename)
    return score_accuracy(items, predictions, False, true_label="correct")


@click.command()
@click.argument('task')
@click.argument('filename')
def main(task, filename):    
    # filename = 'DExperts/test-generations/toxicity/emotion/generations.jsonl'
    items = []

    with jsonlines.open(filename) as f:
        for item in f:
            items.append(item)

    result = {}

    if task == "emotion":
        result["emotion"] = eval_emotion(filename, items)
    elif task == "sentiment":
        result["sentiment"] = eval_sentiment(filename, items)
    elif task == "news-category":
        result["news-category"] = eval_news_category(filename, items)
    elif task == "bbc-news":
        result["bbc-news"] = eval_bbc_news(filename, items)

    result["toxicity"] = eval_toxicity(filename, items)
    result["grammar"] = eval_grammar(filename, items)
    
    df = pd.DataFrame(result)
    print(filename)
    print(df)
    df.to_csv(filename + ".eval.csv")


if __name__ == "__main__":
    main()