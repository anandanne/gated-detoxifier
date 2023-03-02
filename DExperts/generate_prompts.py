from datasets import load_dataset
import jsonlines

NEWS_CATEGORIES = [
    "ENTERTAINMENT", "POLITICS", "WELLNESS", "TRAVEL", "STYLE & BEAUTY",
    "PARENTING", "HEALTHY LIVING", "QUEER VOICES", "FOOD & DRINK", "BUSINESS"
    ]

def generate_news(count=None):
    dataset = load_dataset("heegyu/news-category-balanced-top10", split="train")
    if count:
        dataset = dataset.shuffle(42) #.select(range(count))
    if count:
        out = jsonlines.open(f"prompts/news-{count}.jsonl", "w")
    else:
        out = jsonlines.open(f"prompts/news-all.jsonl", "w")

    for category in NEWS_CATEGORIES:
        i = 0
        for item in dataset:
            if item['category'] != category:
                continue
            
            prompt = f"{item['category']} Title: {item['headline']}\nContent: "
            out.write({
                "prompt": {"text": prompt}
            })
            i += 1
            print(item['category'], category, i)

            if count and count == i:
                break

        print("hi")

    out.close()

def generate_emotion(count_per_emotion):
    # emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    emotions = ["sadness im so sad. ", "joy im so happy. ", "love i feel romantic. ", "anger im furious. ", "fear im so scared. ", "surprise im so surprised. "]
    out = jsonlines.open(f"prompts/emotion-{count_per_emotion}.jsonl", "w")

    for emotion in emotions:
        for item in range(count_per_emotion):
            prompt = emotion
            out.write({
                "prompt": {"text": prompt}
            })

    out.close()

def generate_sentiment(count_per_emotion):
    # emotions = ["positive", "negative"]
    emotions = ["positive it was good. ", "negative it was bad. "]
    out = jsonlines.open(f"prompts/sentiment-{count_per_emotion}.jsonl", "w")

    for emotion in emotions:
        for item in range(count_per_emotion):
            prompt = f"{emotion}"
            out.write({
                "prompt": {"text": prompt}
            })

    out.close()


if __name__ == "__main__":
    for count in [5, 10, 1000]:
        # generate_emotion(count)
        generate_sentiment(count)
    # for count in [10, 1000]:
    #     generate_news(count)
    # for count in [2]:
    #     generate_news(count)