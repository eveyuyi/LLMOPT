import argparse
import json
import socket
import subprocess
import tempfile
import random
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555


def send_request(host: str, port: int, req: dict) -> dict:
    """
    向模型服务发送一次请求，并返回响应 JSON。
    """
    with socket.create_connection((host, port)) as sock:
        file = sock.makefile("rwb")
        file.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        file.flush()

        line = file.readline()
        if not line:
            raise RuntimeError("No response from model server")
        return json.loads(line.decode("utf-8"))


def run_pyomo_code(code_str: str, timeout: int = 60) -> tuple[str, str]:
    """
    将生成的 Pyomo 代码写入临时文件并执行，返回 stdout / stderr。
    """
    temp_id = random.randint(100000, 999999)
    code_path = Path(tempfile.gettempdir()) / f"llmopt_single_{temp_id}.py"
    try:
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code_str)

        proc = subprocess.run(
            ["python", str(code_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        stdout = proc.stdout.decode("utf-8", errors="ignore")
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        return stdout, stderr
    finally:
        if code_path.exists():
            code_path.unlink()


def test_question(host: str, port: int, question: str, execute: bool = False):
    """
    调用模型服务完成 Q2F -> F2C，并可选执行 Pyomo 代码。
    """
    print("发送 Q2F 请求...")
    q2f_resp = send_request(host, port, {"type": "q2f", "payload": {"question": question}})
    if not q2f_resp.get("ok"):
        raise RuntimeError(f"Q2F 失败: {q2f_resp.get('error')}")
    five_elem = q2f_resp.get("result")
    print("five-element:")
    print(five_elem)
    print("-" * 80)

    print("发送 F2C 请求...")
    f2c_resp = send_request(host, port, {"type": "f2c", "payload": {"five_elem": five_elem}})
    if not f2c_resp.get("ok"):
        raise RuntimeError(f"F2C 失败: {f2c_resp.get('error')}")
    code = f2c_resp.get("result")
    print("生成的 Pyomo 代码:")
    print(code)
    print("-" * 80)

    if execute:
        print("执行 Pyomo 代码...")
        stdout, stderr = run_pyomo_code(code)
        print("[STDOUT]")
        print(stdout)
        print("[STDERR]")
        print(stderr or "(empty)")


def main():
    parser = argparse.ArgumentParser(description="调用模型服务测试单个问题")
    parser.add_argument(
        "--question",
        type=str,
        help="要测试的问题文本；可直接传字符串或使用 --question-file",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="包含问题的文本文件路径（只读取第一行）",
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="模型服务 Host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="模型服务端口")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="是否执行生成的 Pyomo 代码",
    )

    args = parser.parse_args()

    if not args.question and not args.question_file:
        parser.error("必须传入 --question 或 --question-file")

    question = args.question
    if args.question_file:
        with open(args.question_file, "r", encoding="utf-8") as f:
            question = f.read().strip()

    if not question:
        parser.error("问题内容为空")

    test_question(args.host, args.port, question, execute=args.execute)


if __name__ == "__main__":
    main()


