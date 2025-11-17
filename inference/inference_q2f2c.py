from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
from pathlib import Path
import sys


# 添加项目根目录到路径，以便导入 prompts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import prompts.generate_prompt as generate_prompt


# load model and tokenizer
path = '/data/models/LLMOPT-Qwen2.5-14B'
path_t = '/data/models/LLMOPT-Qwen2.5-14B'
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(path_t)


# inference to get five elements
def infer_five_elem(question):
    messages = [
        {"role": "user", "content": generate_prompt.Q2F(question)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")

    if "```text" in response:
        return response.split("```text")[1].split("```")[0]
    elif "```plaintext" in response:
        return response.split("```plaintext")[1].split("```")[0]
    elif "```" in response:
        return response.split("```")[1].split("```")[0]
    else:
        return None


# inference to get pyomo python code
def infer_code(five_elem):
    messages = [
        {"role": "user", "content": generate_prompt.F2C(five_elem)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    ans = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return ans.split("```python")[1].split("```")[0].replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')


# execute the code
def test_code(code_str):
    code_path = f"./test.py"
    with open(code_path, "w") as f1:
        f1.write(code_str)

    ans = subprocess.run(f"python {code_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # return answer logs, error code
    return str(ans.stdout.decode('gbk', errors='ignore')), str(ans.stderr.decode('gbk', errors='ignore'))


# example usage
question = "The Li family plans to invest their retirement fund in commercial real estate. Property 1 has an annual income of $12,500, Property 2 has an annual income of $35,000, Property 3 has an annual income of $23,000, and Property 4 has an annual income of $100,000. The decision to be made is whether to buy or not buy each property, not the quantity, as there is only one property per property. Help them decide which properties to purchase to maximize their annual income.\nProperty 1 costs $1.5 million, Property 2 costs $2.1 million, Property 3 costs $2.3 million, and Property 4 costs $4.2 million. The Li family's budget is $7 million.\n\nIf they purchase Property 4, then they cannot purchase Property 3."
five_elem = infer_five_elem(question)
code_str = infer_code(five_elem)
out_log, err_log = test_code(code_str)
print(out_log)
