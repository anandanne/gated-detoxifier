from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import jsonlines
from tqdm.auto import tqdm
from collections import defaultdict
import click


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Classifier:

    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def classify_items(self, items, batch_size: int, id2label):
        predictions = []

        for i in tqdm(range(0, len(items), batch_size)):
            end = min(i + batch_size, len(items))
            batch = items[i:end]
            batch = [x['generation'] for x in batch]
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

    for k in sorted(total.keys()):
        print(k, correct[k] / total[k] ,f"{correct[k]} of {total[k]}")


def eval_news_category(filename, items):
    classifier = Classifier("heegyu/roberta-base-news-category-top10")
    id2label = [
        "ENTERTAINMENT", "POLITICS", "WELLNESS", "TRAVEL", "STYLE",
        "PARENTING", "HEALTHY", "QUEER", "FOOD", "BUSINESS"
        ]
    predictions = classifier.classify_items(items, 8, id2label)

    print("Topic score for", filename)
    score_accuracy(items, predictions)
    return {}


def eval_emotion(filename, items):
    classifier = Classifier("Aron/distilbert-base-uncased-finetuned-emotion")
    id2label = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    predictions = classifier.classify_items(items, 8, id2label)

    print("Sentiment score for", filename)
    score_accuracy(items, predictions)
    return {}

def eval_sentiment(filename, items):
    classifier = Classifier("wrmurray/roberta-base-finetuned-imdb")
    id2label = ["negative", "positive"]
    predictions = classifier.classify_items(items, 8, id2label)

    print("Sentiment score for", filename)
    score_accuracy(items, predictions)
    return {}

def eval_toxicity(filename, items):
    classifier = Classifier("s-nlp/roberta_toxicity_classifier")
    id2label = ["non-toxic", "toxic"]
    predictions = classifier.classify_items(items, 8, id2label)

    print("Toxicity score for", filename)
    score_accuracy(items, predictions, False, true_label="toxic")
    return {}

def eval_grammar(filename, items):
    classifier = Classifier("cointegrated/roberta-large-cola-krishna2020")
    id2label = ["correct", "error"]
    predictions = classifier.classify_items(items, 8, id2label)

    print("Grammar score for", filename)
    score_accuracy(items, predictions, False, true_label="correct")
    return {}


@click.command()
@click.argument('task')
@click.argument('filename')
def main(task, filename):    
    # filename = 'DExperts/test-generations/toxicity/emotion/generations.jsonl'
    items = []

    with jsonlines.open(filename) as f:
        for item in f:
            items.append(item)

    if task == "emotion":
        eval_emotion(filename, items)
    elif task == "sentiment":
        eval_sentiment(filename, items)
    elif task == "news-category":
        eval_news_category(filename, items)

    eval_toxicity(filename, items)
    eval_grammar(filename, items)


if __name__ == "__main__":
    main()