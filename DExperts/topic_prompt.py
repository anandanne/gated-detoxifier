from datasets import load_dataset
import jsonlines


dataset = load_dataset("heegyu/news-category-balanced-top10", split="train")

out = jsonlines.open("prompts/news.jsonl", "w")

for item in dataset:
    prompt = f"{item['category']} Title: {item['headline']}\nContent: "
    out.write({
        "prompt": {"text": prompt}
    })

out.close()
