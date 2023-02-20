from datasets import load_dataset
import jsonlines


def generate_news(count=None):
    dataset = load_dataset("heegyu/news-category-balanced-top10", split="train")
    if count:
        dataset = dataset.shuffle(42).select(range(count))
    out = jsonlines.open(f"prompts/news-{count}.jsonl", "w")


    for item in dataset:
        prompt = f"{item['category']} Title: {item['headline']}\nContent: "
        out.write({
            "prompt": {"text": prompt}
        })

    out.close()

def generate_emotion(count_per_emotion):
    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    out = jsonlines.open(f"prompts/emotion-{count_per_emotion}.jsonl", "w")

    for emotion in emotions:
        for item in range(count_per_emotion):
            prompt = f"{emotion} "
            out.write({
                "prompt": {"text": prompt}
            })

    out.close()

def generate_sentiment(count_per_emotion):
    emotions = ["positive", "negative"]
    out = jsonlines.open(f"prompts/sentiment-{count_per_emotion}.jsonl", "w")

    for emotion in emotions:
        for item in range(count_per_emotion):
            prompt = f"{emotion} "
            out.write({
                "prompt": {"text": prompt}
            })

    out.close()


if __name__ == "__main__":
    # for count in [5, 250]:
    #     generate_emotion(count)
    #     generate_sentiment(count)
    for count in [100, None]:
        generate_news(count)