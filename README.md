# gated-detoxifier

절차
1. DExperts로 topic, emotion, sentiment 생성
2. GeDi로 topic, emotion, sentiment 생성
3. classifier로 분석

### Prompt Format

jsonl line
```json
{
    "prompt": {"text": "anger "},
}
```

### Generate Format

jsonl line
```json
{
    "prompt": "?",
    "detoxifier": "gedi", # or "dexperts",
    
}
```