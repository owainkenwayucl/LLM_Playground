from openai import OpenAI
import sys
import copy
import time

import logging
import configparser

# Read in configuration
config = configparser.ConfigParser()
config.read("llm.ini")

logging.disable(logging.WARNING)
bold_on = "\033[1m"
style_off = "\033[0m"

# Set up LLM endpoint
endpoint = config["OPENAI"]["endpoint"]
model = config["OPENAI"]["model"]
api_key = config["OPENAI"]["api_key"]
print(f"{bold_on}Starting up - LLM endpoint = {style_off}{endpoint}/{model}")


client = OpenAI(
    base_url=endpoint,
    api_key=api_key,
)


messages_ = [
]

messages = copy.deepcopy(messages_)
avatar = "🤖"

reasoning = False

while True:
    line = input("? ")
    if 'bye' == line.strip().lower():
        sys.exit()

    if 'reprogram' == line.strip().lower():
        print(f"Current prompt: {messages_}")
        print(f"Current avatar: {avatar}")

        new_prompt = input("New prompt: ")
        avatar = input("New avatar: ")

        messages_ = [
            {"role": "system", "content": new_prompt},
        ]
        messages = copy.deepcopy(messages_)
        continue
        
    if 'inspect' == line.strip().lower():
        print(f"Model: {checkpoint_name}")
        print(f"Chat state: {messages}")
        print(f"Avatar: {avatar}")
        print(f"Reasoning: {reasoning}")
        continue

    if 'reasoning on' == line.strip().lower():
        print(f"Enabling reasoning.")
        reasoning=True
        continue

    if 'reasoning off' == line.strip().lower():
        print(f"Disabling reasoning.")
        reasoning=False
        continue

    if 'forget' == line.strip().lower():
        messages = copy.deepcopy(messages_)
        continue

    t_start = time.time()
    line = line.strip()
    messages.append({"role":"user","content":line})

    response = client.responses.create(
        model = model,
        messages=messages
    ).output_text

    t_elapsed = time.time() - t_start
    print(f"{bold_on}---\n{avatar} :{style_off} {response}\n{bold_on}--- [{t_elapsed} seconds] {style_off}\n")

    messages.append({"role":"assistant","content":response})

