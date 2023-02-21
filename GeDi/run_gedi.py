import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any
import json
from typing import TypeVar, Iterable, List, Union, Any
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import click
import jsonlines
import pandas as pd

from modeling_gpt2 import GPT2LMHeadModel

from transformers import (
    GPT2Config,
    GPT2Tokenizer
)

T = TypeVar('T')


ALLOWED_MODELS = ['gpt3', 'gpt2', 'dexperts', 'dexperts-gpt3', 'pplm']
torch.set_grad_enabled(False)



def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        # if response['response']:
        #     response = unpack_scores(response['response'])[0]
        # else:
        #     response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {'text': generation, **response}


def collate(dataset: Optional[pd.DataFrame], generations: List[str], responses: Iterable[Dict[str, Any]], output_file: str):
    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        print(f"Detected samples per prompt:", n)
        generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
        dataset['generations'] = generations_col

    dataset.to_json(output_file, orient='records', lines=True)


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--use-eos/--use-dataset', default=False, help='Whether to use EOS or a dataset file for generation.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=32, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=32)
@click.option('--gedi/--no-gedi', default=True)
@click.option('--resume/--no-resume', default=False)
@click.option('--filter_p', default=0.8, type=float, help='1 - rho from paper, should be between 0 and 1 higher filter_p means more aggressive topic steering')
@click.option('--target_p', default=0.8, type=float, help='tau from paper, preserves tokens that are classified as correct topic')
@click.option('--disc_weight', default=30, type=int, help='omega from paper, higher disc_weight means more aggressive topic steering (30)')
@click.option('--p', default=1.0, type=float, help='Hyperparameter for nucleus sampling')
def main(output_dir: str, dataset_file: Optional[str], use_eos: bool, model: str, gedi: bool,
         target_p: float, disc_weight: int, n: int, max_tokens: int, batch_size: int, resume: bool, filter_p: float, p: float):
    # Load prompts
    if dataset_file:
        assert not use_eos
        # Load prompts from dataset file
        assert dataset_file.endswith('.jsonl')
        dataset = pd.read_json(dataset_file, lines=True)
        prompts = pd.json_normalize(dataset['prompt'])['text']
    elif use_eos:
        assert not dataset_file
        dataset = None
        # Create EOS prompts
        if model_type in ['gpt2', 'gpt2-affect', 'gpt2-ensemble', 'gpt2-naughty-list', 'pplm']:
            prompts = pd.Series('<|endoftext|>')
        elif model_type == 'ctrl':
            # HACK: update gen_samples since we use it as our batch size for pipelines
            prompts = pd.Series('').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'gpt3':
            prompts = pd.Series('').repeat(n // batch_size + 1)
        else:
            raise RuntimeError('Model not implemented with EOS prompts')
    else:
        raise click.exceptions.MissingParameter('Missing --dataset-file or --use-eos option.')
    print('Prompts:', '\n', prompts)

    # Create output files
    output_dir = Path(output_dir)
    generations_file = output_dir / 'generations.jsonl'
    assert resume or not os.path.exists(generations_file)   # don't overwrite generations!
    ensure_dir(output_dir)

    mode = "detoxifier"
    code_desired = "dirty"
    code_undesired = "clean"
    model_type = 'gpt2'
    gen_type = "gedi"
    gen_model_name_or_path = "gpt2" # "gpt2-medium"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(gen_model_name_or_path, do_lower_case=False)
    model = GPT2LMHeadModel.from_pretrained(gen_model_name_or_path)#, load_in_half_prec=True)
    model = model.to(device).eval()

    gedi_model_name_or_path = 'pretrained_models/gedi_detoxifier'
    gedi_model = GPT2LMHeadModel.from_pretrained(gedi_model_name_or_path).eval().to(device)
    disc_weight = 30 # #omega from paper, higher disc_weight means more aggressive topic steering (30)
    filter_p = 0.8 #1 - rho from paper, should be between 0 and 1 higher filter_p means more aggressive topic steering
    target_p = 0.8 #tau from paper, preserves tokens that are classified as correct topic
    class_bias = 0 #hyperparameter that determines class prior, set to uniform by default

    print(gedi)
    def generate_text(prompt):
        text_ids = tokenizer.encode(prompt)
        encoded_prompts=torch.LongTensor(text_ids).unsqueeze(0).to(device)
        if encoded_prompts.shape[1] > 512:
            encoded_prompts = encoded_prompts[:, -512:]

        # multi_code = tokenizer.encode(secondary_code)
        attr_class = 1

        generated_sequence = model.generate(
            input_ids=encoded_prompts,
            pad_lens=None,
            max_length=min(1024, encoded_prompts.shape[1] + max_tokens),
            min_length=min(1024, encoded_prompts.shape[1] + max_tokens),
            top_k=None,
            top_p=1.0,
            # repetition_penalty= 1.2,
            # rep_penalty_scale= 10,
            eos_token_ids = tokenizer.eos_token_id,
            pad_token_id = 0,
            do_sample= True,
            penalize_cond= True,
            gedi_model= gedi_model if gedi else None,
            tokenizer= tokenizer,
            disc_weight= disc_weight,
            filter_p = filter_p,
            target_p = target_p,
            class_bias = class_bias,
            attr_class = attr_class,
            code_0 = code_desired,
            code_1 = code_undesired,
            multi_code=None,
            num_return_sequences=n
            )

        texts = [tokenizer.decode(output, skip_special_tokens=True)[len(prompt):] for output in generated_sequence.tolist()]
        # texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_sequence.tolist()]
        return texts

    # prompt = "It is really "
    # print(generate_text(prompt))

    with jsonlines.open(generations_file, "w") as f:
        for p in tqdm(prompts):
            generations = generate_text(p)
            for g in generations:
                f.write({
                    "prompt": p,
                    "generation": g
                })

    # Generate and collate perspective scores
    # generations = []
    # for i, gen in enumerate(generations_iter):
    #     generations.append(gen)
        # perspective(f'generation-{i}', gen)

    torch.cuda.empty_cache()
    # perspective.stop()
    print('Finished generation and perspective scoring!')

    # if os.path.exists(perspective_file):
    #     print('Collating output files')
    #     collate(dataset, generations, load_jsonl(perspective_file), output_file)


if __name__ == '__main__':
    main()
