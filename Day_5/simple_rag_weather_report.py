from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator

import json
import requests
import os
import re

from dotenv import load_dotenv
load_dotenv("../../dotenv.sh")
OWM_API_KEY = os.getenv("OWM_API_KEY")

device = Accelerator().device

# 1. Define the actual local tool function
def get_weather(city: str) -> dict:
    import requests
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    API_KEY = OWM_API_KEY
    QUERY_STRING = f"?q={city}&units=metric&APPID={API_KEY}"
    URL = BASE_URL + QUERY_STRING

    response = requests.get(URL)
    if response.ok:
        return response.json()
    else:
        return {"error": "Could not fetch weather data."}

def calculate(expression: str) -> str:
    try:
        # Evaluate the mathematical expression safely
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


tools_dict = {
    "get_weather": get_weather,
    "calculate": calculate
}


# 2. Setup model and tokenizer
checkpoint = "HuggingFaceTB/SmolLM3-3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, 
                                             dtype=torch.bfloat16).to(device)

# 3. Define the tool schema and starting conversation
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather info for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Calculate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
]



messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the current weather in Chennai?"}
]

# ==========================================
# STEP 1: Generate the Tool Call
# ==========================================
inputs = tokenizer.apply_chat_template(
    messages, 
    xml_tools=tools,
    add_generation_prompt=True,
    #tokenize=False, 
    return_tensors="pt"
).to(device)

r = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True)
assistant_text = tokenizer.decode(outputs[0][len(inputs[0]):], 
                                  skip_special_tokens=True).strip()

print(f"--- Model Output ---\n{assistant_text}\n--------------------------------")

# Append model's response to conversation history
messages.append({"role": "assistant", "content": assistant_text})

# ==========================================
# STEP 2: Parse and Execute the Tool
# ==========================================
# Extract contents inside <tool_call>...</tool_call>
match = re.search(r"<tool_call>(.*?)</tool_call>", assistant_text, re.DOTALL)

if match:
    tool_call_data = json.loads(match.group(1).strip())
    tool_name = tool_call_data.get("name")
    tool_args = tool_call_data.get("arguments", {})

    print(f"--- Executing Tool: {tool_name} with args {tool_args} ---")

    tool_function = tools_dict.get(tool_name)
    try:
        tool_result = tool_function(**tool_args) if tool_function else None
    except Exception as e:
        tool_result = f"Error: {e}"
    else:
        if not tool_function:
            tool_result = "Error: Tool not found."

    print(f"Tool Result: {tool_result}\n")

    # Append the result back using the 'tool' role
    messages.append({
        "role": "tool",
        "name": tool_name,
        "content": tool_result
    })

    print(f"--- Updated Messages ---\n{json.dumps(messages, indent=2)}\n--------------------------------")

    # ==========================================
    # STEP 3: Generate Final Answer
    # ==========================================
    final_inputs = tokenizer.apply_chat_template(
        messages, 
        #xml_tools=tools, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    final_outputs = model.generate(**final_inputs, max_new_tokens=512, do_sample=True)
    final_answer = tokenizer.decode(final_outputs[0][len(final_inputs[0]):], skip_special_tokens=True)

    print(f"--- Final Answer ---\n{final_answer.strip()}")
else:
    print("Model did not request a tool call.")
