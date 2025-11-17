from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
import sys
from typing import Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM


# 将项目根目录加入 sys.path，方便导入 prompts
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import prompts.generate_prompt as generate_prompt  # noqa: E402


HOST = "0.0.0.0"
PORT = 5555  # 如需在同一机器上启动多个服务，可改这个端口


def load_model(
    model_path: str,
    tokenizer_path: str | None = None,
    device: str = "cuda",
):
    """
    加载模型和 tokenizer。只在服务启动时调用一次。
    """
    if tokenizer_path is None:
        tokenizer_path = model_path

    print(f"[model_server] Loading model from: {model_path}")
    print(f"[model_server] Loading tokenizer from: {tokenizer_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    print("[model_server] Model and tokenizer loaded.")
    return model, tokenizer, device


def infer_five_elem(model, tokenizer, device: str, question: str) -> str | None:
    messages = [
        {"role": "user", "content": generate_prompt.Q2F(question)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    response = (
        response
        .replace("\\n", "\n")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("\\\"", "\"")
    )

    if "```text" in response:
        return response.split("```text")[1].split("```")[0]
    elif "```plaintext" in response:
        return response.split("```plaintext")[1].split("```")[0]
    elif "```" in response:
        return response.split("```")[1].split("```")[0]
    else:
        return None


def infer_code(model, tokenizer, device: str, five_elem: str) -> str:
    messages = [
        {"role": "user", "content": generate_prompt.F2C(five_elem)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    ans = (
        response
        .replace("\\n", "\n")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("\\\"", "\"")
    )

    # 提取 Python 代码
    if "```python" in ans:
        code = ans.split("```python")[1].split("```")[0]
    elif "```" in ans:
        code = ans.split("```")[1].split("```")[0]
    else:
        code = ans

    return (
        code
        .replace('print("\\\n', 'print("')
        .replace('print(f"\\\n', 'print(f"')
    )


def handle_request(
    conn: socket.socket,
    addr,
    model,
    tokenizer,
    device: str,
):
    """
    处理单个 TCP 连接。协议：一行一个 JSON，请求/响应皆为 JSON Lines。
    """
    with conn:
        try:
            file = conn.makefile("rwb")
            for line in file:
                if not line:
                    break
                try:
                    req = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as e:
                    resp = {"ok": False, "error": f"JSONDecodeError: {e}"}
                    file.write((json.dumps(resp) + "\n").encode("utf-8"))
                    file.flush()
                    continue

                req_type = req.get("type")
                payload: Dict[str, Any] = req.get("payload", {})

                try:
                    if req_type == "q2f":
                        question = payload.get("question", "")
                        result = infer_five_elem(model, tokenizer, device, question)
                        resp = {"ok": True, "result": result}
                    elif req_type == "f2c":
                        five_elem = payload.get("five_elem", "")
                        result = infer_code(model, tokenizer, device, five_elem)
                        resp = {"ok": True, "result": result}
                    else:
                        resp = {"ok": False, "error": f"Unknown type: {req_type}"}
                except Exception as e:  # noqa: BLE001
                    resp = {"ok": False, "error": repr(e)}

                file.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                file.flush()
        except Exception as e:  # noqa: BLE001
            print(f"[model_server] Connection error from {addr}: {e}")


def serve(model_path: str, tokenizer_path: str | None = None, device: str = "cuda"):
    model, tokenizer, device = load_model(model_path, tokenizer_path, device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"[model_server] Listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            print(f"[model_server] Accepted connection from {addr}")
            t = threading.Thread(
                target=handle_request,
                args=(conn, addr, model, tokenizer, device),
                daemon=True,
            )
            t.start()


if __name__ == "__main__":
    # 可以根据需要改成从环境变量或命令行参数读取
    default_model_path = "/data/models/LLMOPT-Qwen2.5-14B"
    default_tokenizer_path = "/data/models/LLMOPT-Qwen2.5-14B"
    serve(default_model_path, default_tokenizer_path, device="cuda")


