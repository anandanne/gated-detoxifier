
import click
import pandas as pd
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import torch
from torch import FloatTensor, LongTensor
from tqdm import tqdm
import os
import jsonlines
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2PreTrainedModel, LogitsProcessorList, LogitsProcessor, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer

ALLOWED_MODELS = ['gpt3', 'gpt2', 'dexperts', 'pplm', 'gedi']
ALLOWED_PROMPT = ["yelp", "emotion", "bbc-news"]
PROMPT = {
    "yelp": ["topic: positive\n", "topic: negative\n"],
    "emotion": [f"topic: {k}\n" for k in ["sadness", "joy", "love", "anger", "fear", "surprise"]],
    "bbc-news": [f"topic: {k}\n" for k in ["tech", "business", "sport", "entertainment", "politics"]],
}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

class GPT2LMHeadModelForGate(GPT2LMHeadModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.ctg_warper = None

        # print(config)

    def _get_logits_warper(self, *args, **kwargs) -> LogitsProcessorList:
        # print(kwargs)
        # print("top-k", self.config.top_k)
        # top-k가 기본값으로 50이 되버린다;;;
        # self.config.top_k = None
        warpers = super()._get_logits_warper(*args, **kwargs)
        if self.ctg_warper is not None:
            self.ctg_warper.original_warpers = warpers
            return self.ctg_warper
        else:
            return warpers

@dataclass
class GPT2Generator:
    model_name: str
    num_return_sequences: int
    max_tokens: int
    p: float = 1.0

    def __post_init__(self):
        self.model = GPT2LMHeadModelForGate.from_pretrained(self.model_name).to(device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            num_return_sequences=self.num_return_sequences,
            max_new_tokens=self.max_tokens,
            top_p=self.p,
            do_sample=True,
            early_stopping=True,
            pad_token_id=50256
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [x[len(prompt):] for x in outputs]
        return outputs

class GatedLogitsProcessor(LogitsProcessor):
    def __init__(self, generation_tokenizer, classifier_tokenizer, classifier, post_processor: LogitsProcessor, label_index: int = 1) -> None:
        super().__init__()
        self.post_processor = post_processor
        self.original_warpers = None
        self.generation_tokenizer = generation_tokenizer
        self.classifier_tokenizer = classifier_tokenizer
        self.classifier = classifier
        self.label_index = label_index

    def _classify_text(self, texts):
        inputs = self.classifier_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = self.classifier(**inputs).logits.softmax(-1)[:, self.label_index]
        outputs = (outputs > 0.5).float()
        return outputs
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor: # of shape (batch_size, config.vocab_size)
        warped_scores = self.original_warpers(input_ids, scores)
        next_tokens = torch.multinomial(warped_scores.softmax(-1), num_samples=1) # (b, 1)
        current_ids = torch.cat([input_ids, next_tokens], 1) # (b, s + 1)
        current_texts = self.generation_tokenizer.batch_decode(current_ids, skip_special_tokens=True)
        gate_output = self._classify_text(current_texts)
        

        logits = torch.ones_like(scores, device=scores.device) * -50000
        logits[torch.range(0, logits.shape[0] - 1, device=scores.device, dtype=torch.long), next_tokens.long().squeeze(1)] = 0

        if gate_output.sum().item() > 0:
            print(current_texts)
            print("toxic!", gate_output.cpu())
            gate_output = gate_output.unsqueeze(1)

            guided = self.post_processor(input_ids, scores)
            gated_scores = logits * (1 - gate_output) + gate_output * guided
            return self.original_warpers(input_ids, gated_scores)
        else:
            return logits
        
class DExpertLogitsProcessor(LogitsProcessor):

    def __init__(self, alpha, expert, anti_expert) -> None:
        super().__init__()
        self.alpha = alpha
        self.expert = expert
        self.anti_expert = anti_expert

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor: # of shape (batch_size, config.vocab_size)
        expert_logits = self.expert(input_ids).logits[:, -1]
        anti_expert_logits = self.anti_expert(input_ids).logits[:, -1]
        return scores + self.alpha * (expert_logits - anti_expert_logits)


@dataclass
class DExpertGenerator:
    model_name: str
    expert_model_name: str
    anti_expert_model_name: str
    num_return_sequences: int
    max_tokens: int
    p: float = 1.0
    alpha: float = 1.0
    classifier_model_name: Optional[str] = None

    def __post_init__(self):
        self.model = GPT2LMHeadModelForGate.from_pretrained(self.model_name).to(device).eval()
        self.expert = GPT2LMHeadModel.from_pretrained(self.expert_model_name).to(device).eval()
        self.anti_expert = GPT2LMHeadModel.from_pretrained(self.anti_expert_model_name).to(device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        processor = DExpertLogitsProcessor(
            alpha=self.alpha,
            expert=self.expert,
            anti_expert=self.anti_expert
        )
        if self.classifier_model_name and self.classifier_model_name != "no":
            processor = GatedLogitsProcessor(
                self.tokenizer,
                AutoTokenizer.from_pretrained(self.classifier_model_name),
                AutoModelForSequenceClassification.from_pretrained(self.classifier_model_name).to(device).eval(),
                processor
            )

        # self.logits_processors = LogitsProcessorList([processor])
        self.model.ctg_warper = processor
    
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        outputs = self.model.generate(
            **inputs,
            num_return_sequences=self.num_return_sequences,
            max_new_tokens=self.max_tokens,
            top_p=self.p,
            do_sample=True,
            early_stopping=True,
            pad_token_id=50256
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [x[len(prompt):] for x in outputs]
        return outputs
    
class GeDiLogitsProcessor(LogitsProcessor):

    def __init__(self, disc_weight, logits_scale, expert, anti_expert) -> None:
        super().__init__()
        self.disc_weight = disc_weight
        self.logits_scale = logits_scale
        self.expert = expert
        self.anti_expert = anti_expert

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor: # of shape (batch_size, config.vocab_size)
        expert_logits = self.expert(input_ids).logits[:, -1, :].softmax(-1)
        anti_expert_logits = self.anti_expert(input_ids).logits[:, -1, :].softmax(-1)
        desired_logits = expert_logits / (expert_logits + anti_expert_logits)
        desired_logits = desired_logits.log_softmax(-1)

        return scores + self.disc_weight * desired_logits

@dataclass
class GeDiGenerator:
    model_name: str
    expert_model_name: str
    anti_expert_model_name: str
    num_return_sequences: int
    max_tokens: int
    disc_weight: float
    logits_scale: float
    classifier_model_name: Optional[str] = None
    p: float = 1.0

    def __post_init__(self):
        self.model = GPT2LMHeadModelForGate.from_pretrained(self.model_name).to(device).eval()
        self.expert = GPT2LMHeadModel.from_pretrained(self.expert_model_name).to(device).eval()
        self.anti_expert = GPT2LMHeadModel.from_pretrained(self.anti_expert_model_name).to(device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        processor = GeDiLogitsProcessor(
            logits_scale=self.logits_scale,
            disc_weight=self.disc_weight,
            expert=self.expert,
            anti_expert=self.anti_expert
        )
        if self.classifier_model_name and self.classifier_model_name != "no":
            processor = GatedLogitsProcessor(
                self.tokenizer,
                AutoTokenizer.from_pretrained(self.classifier_model_name),
                AutoModelForSequenceClassification.from_pretrained(self.classifier_model_name).to(device).eval(),
                processor
            )
        self.model.ctg_warper = processor
    
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        outputs = self.model.generate(
            **inputs,
            num_return_sequences=self.num_return_sequences,
            max_new_tokens=self.max_tokens,
            top_p=self.p,
            do_sample=True,
            early_stopping=True,
            pad_token_id=50256
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [x[len(prompt):] for x in outputs]
        return outputs
    
        
        


@click.command()
@click.argument('output-file')
@click.option('--prompt', required=True, type=click.Choice(ALLOWED_PROMPT))
@click.option('--use-eos/--use-dataset', default=False, help='Whether to use EOS or a dataset file for generation.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--toxic-model', type=str, default=None, help='Anti-expert for DExperts')
@click.option('--nontoxic-model', type=str, default=None, help='Expert for DExperts')
@click.option('--classifier-model', type=str, default=None, help='Classifier for Gated Detoxifier')
@click.option('--perspective-rate-limit', default=25)
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=32, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=8)
@click.option('--resume/--no-resume', default=False)
@click.option('--overwrite/--no-overwrite', default=False)
@click.option('--alpha', default=0.0, help='Hyperparameter for dexperts')
@click.option('--filter_p', default=0.9, type=float, help='Hyperparameter for truncation of p_base')
@click.option('--disc_weight', default=15, type=float, help='GeDi omega')
@click.option('--logits_scale', default=10.0, type=float, help='GeDi logits scale')
@click.option('--p', default=1.0, type=float, help='Hyperparameter for nucleus sampling')
def main(output_file: str, prompt: str, use_eos: bool, model: str, model_type: str, nontoxic_model: str,
         toxic_model: str, perspective_rate_limit: int, n: int, max_tokens: int, batch_size: int, 
         resume: bool, overwrite: bool,
         disc_weight: float, logits_scale: float,
         classifier_model: str, alpha: float, filter_p: float, p: float):
    
    assert resume or overwrite or not os.path.exists(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    if model_type == "gpt2":
        generator = GPT2Generator(
            model_name=model,
            num_return_sequences=batch_size,
            max_tokens=max_tokens,
            p=p
            )
    elif model_type == "dexperts":
        generator = DExpertGenerator(
            model_name=model,
            num_return_sequences=batch_size,
            max_tokens=max_tokens,
            alpha=alpha,
            p=p,
            expert_model_name=nontoxic_model,
            anti_expert_model_name=toxic_model,
            classifier_model_name=classifier_model
            )
    elif model_type == "gedi":
        generator = GeDiGenerator(
            model_name=model,
            num_return_sequences=batch_size,
            max_tokens=max_tokens,
            disc_weight=disc_weight,
            logits_scale=logits_scale,
            p=p,
            expert_model_name=nontoxic_model,
            anti_expert_model_name=toxic_model,
            classifier_model_name=classifier_model
            )
        
    prompts = PROMPT[prompt]
    fout = jsonlines.open(output_file, "a" if resume else "w")

    progress = tqdm(total=len(prompts) * n, desc=output_file)

    for prompt in prompts:
        for _ in range(n):
            gens = generator.generate(prompt)
            for g in gens:
                fout.write({
                    "prompt": prompt,
                    "generation": g
                })

            progress.update(1)
    
    fout.close()

if __name__ == "__main__":
    main()