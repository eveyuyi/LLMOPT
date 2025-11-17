import json
import os
import socket
import subprocess
import tempfile
import random
from pathlib import Path
from typing import Optional


HOST = "127.0.0.1"
PORT = 5555


def send_request(req: dict) -> dict:
    """
    向常驻模型服务发送一次请求，并返回响应 JSON。
    """
    with socket.create_connection((HOST, PORT)) as sock:
        file = sock.makefile("rwb")
        file.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        file.flush()

        line = file.readline()
        if not line:
            raise RuntimeError("No response from model server")
        return json.loads(line.decode("utf-8"))


def test_one_question(question: str) -> dict:
    """
    通过服务完成 Q2F -> F2C 两步推理，返回 five_elem 和 code。
    """
    # Q2F
    q2f_resp = send_request(
        {"type": "q2f", "payload": {"question": question}}
    )
    if not q2f_resp.get("ok"):
        return {"ok": False, "stage": "q2f", "error": q2f_resp.get("error")}
    five_elem = q2f_resp.get("result")

    # F2C
    f2c_resp = send_request(
        {"type": "f2c", "payload": {"five_elem": five_elem}}
    )
    if not f2c_resp.get("ok"):
        return {
            "ok": False,
            "stage": "f2c",
            "error": f2c_resp.get("error"),
            "five_elem": five_elem,
        }

    return {
        "ok": True,
        "five_elem": five_elem,
        "code": f2c_resp.get("result"),
    }


def run_pyomo_code(code_str: str, timeout: int = 60) -> tuple[str, str]:
    """
    将生成的 Pyomo 代码写入临时文件并执行，返回 stdout / stderr。
    """
    temp_id = random.randint(100000, 999999)
    temp_dir = Path(tempfile.gettempdir())
    code_path = temp_dir / f"llmopt_test_{temp_id}.py"

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


def batch_from_jsonl(
    jsonl_path: str,
    output_path: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """
    使用常驻模型服务批量测试 jsonl 的前 N 条。
    只负责和服务通信，本身不加载模型。
    """
    results = []

    jsonl_path = str(Path(jsonl_path))
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fout = open(output_path, "w", encoding="utf-8")
    else:
        fout = None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if max_samples is not None and len(results) >= max_samples:
                print(f"达到最大数量 {max_samples}，停止测试")
                break
            if not line.strip():
                continue

            data = json.loads(line.strip())
            question = data.get("question", "")
            if not question:
                print(f"跳过第 {idx} 行：问题为空")
                continue

            print(f"[{idx}] 发送问题到模型服务...")
            print(
                f"  问题预览: {question[:80]}..."
                if len(question) > 80
                else f"  问题: {question}"
            )

            result = {
                "index": data.get("index", idx),
                "question": question,
                "ground_truth": data.get("answer", None),
            }

            try:
                resp = test_one_question(question)
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ 调用服务失败: {e}")
                result.update({
                    "ok": False,
                    "error": str(e),
                })
            else:
                result.update(resp)
                if not resp.get("ok"):
                    print(f"  ✗ 推理失败: {resp.get('error')}")
                else:
                    print("  ✓ 推理成功，执行 Pyomo 代码...")
                    if resp.get("code"):
                        stdout, stderr = run_pyomo_code(resp["code"])
                        result["execution_stdout"] = stdout
                        result["execution_stderr"] = stderr or None
                        if stderr:
                            print("  ⚠️  执行有报错（详见结果文件）")
                        else:
                            print("  ✓ 执行成功")
                    else:
                        print("  ⚠️  无代码可执行")

            results.append(result)
            if fout is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
            print()

    if fout is not None:
        fout.close()
        print(f"结果已写入: {output_path}")

    return results


if __name__ == "__main__":
    # 示例：使用 industryor 测试集，只跑前 3 个问题
    project_root = Path(__file__).parent.parent
    test_file = project_root / "data/testset/industryor.jsonl"
    output_file = project_root / "inference/results_server_industryor.jsonl"

    batch_from_jsonl(str(test_file), str(output_file), max_samples=50)


