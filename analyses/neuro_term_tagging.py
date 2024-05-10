import re
import tqdm
import argparse
import asyncio
import aiofiles  # async file I/O

import openai
from openai import AsyncAzureOpenAI  # Verify this supports async requests or replace
from decouple import config as env_config


async def load_existing_results(fname):
    results = {}
    try:
        async with aiofiles.open(f"{results_dir}/{fname}__neuro_term_tagging_results.txt", "r") as f:
            async for line in f:
                word, result = line.strip().split(", ")
                results[word] = result
    except FileNotFoundError:
        pass
    return results


def does_not_contain_alphabet(word):
    return not bool(re.search(r"[a-zA-Z]", word))


async def is_neuroscience_term(word):
    user_prompt = f"Is '{word}' a term commonly seen in neuroscience? Answer with 'yes' or 'no', no additional words."

    client = AsyncAzureOpenAI(
        api_key=env_config("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=env_config("AZURE_OPENAI_ENDPOINT")
    )

    try:
        completion = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": user_prompt.format(word),
                },
            ],
        )
        response = completion.choices[0].message.content.lower()
        cost = completion.usage.prompt_tokens * costdict["gpt-4"]["prompt_tokens"] \
            + completion.usage.completion_tokens * costdict["gpt-4"]["completion_tokens"]
        cost /= 1000

    except openai.BadRequestError:  # got filtered.
        print(f"Error with word: {word}")
        return 'filtered', 0
    
    if re.match(r"yes|no", response):
        return response, cost
    else:
        return None, cost


async def main(fname):
    existing_results = await load_existing_results(fname)
    total_cost = 0

    async with aiofiles.open(f"{data_dir}/{fname}.txt", "r") as f:
        lines = await f.readlines()
        for line in tqdm.tqdm(lines):
            word = line.strip()
            if does_not_contain_alphabet(word):
                print(f"{word} is not a valid word")
                continue

            if word in existing_results:
                print(f"{word} already checked")
                continue

            result, cost = await is_neuroscience_term(word)
            total_cost += cost
            async with aiofiles.open(f"{results_dir}/{fname}__neuro_term_tagging_results.txt", "a") as f:
                print(f"{word}, {result}, total cost: {total_cost}")
                await f.write(f"{word}, {result}\n")


if __name__ == "__main__":
    costdict = {
        "gpt-4": {
            "prompt_tokens": 0.03,
            "completion_tokens": 0.06,  # per 1k tokens
        }
    }
    data_dir = "data"
    results_dir = "token_results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, help="File name to read from")
    args = parser.parse_args()
    # fname = "pretrain_filtered"
    # fname = "neuro_filtered"
    asyncio.run(main(args.fname))
