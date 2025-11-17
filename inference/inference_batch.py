from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import json
import os
import sys
from pathlib import Path

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
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(path_t, trust_remote_code=True)


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
    
    # 提取 Python 代码
    if "```python" in ans:
        code = ans.split("```python")[1].split("```")[0]
    elif "```" in ans:
        # 如果没有 python 标记，尝试提取第一个代码块
        code = ans.split("```")[1].split("```")[0]
    else:
        # 如果没有代码块标记，返回整个响应
        code = ans
    
    return code.replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')


# execute the code
def test_code(code_str):
    # 使用临时文件，避免冲突
    import tempfile
    import random
    temp_id = random.randint(100000, 999999)
    code_path = f"./test_{temp_id}.py"
    try:
        with open(code_path, "w", encoding='utf-8') as f1:
            f1.write(code_str)

        ans = subprocess.run(f"python {code_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

        # return answer logs, error code
        return str(ans.stdout.decode('utf-8', errors='ignore')), str(ans.stderr.decode('utf-8', errors='ignore'))
    finally:
        # 清理临时文件
        if os.path.exists(code_path):
            os.remove(code_path)


# 批量测试函数
def batch_test(jsonl_path, output_path, max_samples=None):
    """
    批量测试问题
    jsonl_path: 测试集文件路径，每行是一个 JSON，包含 "question" 字段
    output_path: 可选，结果输出路径
    """
    results = []
    total = 0
    
    # 统计总行数
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total = sum(1 for line in f if line.strip())
    
    print(f"开始批量测试，共 {total} 个问题...")
    print(f"测试集文件: {jsonl_path}")
    print(f"结果将保存到: {output_path}\n")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if max_samples is not None and len(results) >= max_samples:
                print(f"达到最大数量 {max_samples}，停止测试")
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                question = data.get('question', '')
                if not question:
                    print(f"跳过第 {idx} 行：问题为空")
                    continue
                
                print(f"[{idx}/{total}] 处理问题...")
                print(f"  问题预览: {question[:80]}..." if len(question) > 80 else f"  问题: {question}")
                
                # 推理流程
                five_elem = infer_five_elem(question)
                if five_elem is None:
                    print(f"  ⚠️  警告: 无法提取 five_elem")
                    results.append({
                        'index': data.get('index', idx),
                        'question': question,
                        'five_elem': None,
                        'code': None,
                        'output': None,
                        'error': 'Failed to extract five_elem',
                        'ground_truth': data.get('answer', None)
                    })
                    continue
                
                print(f"  ✓ 已提取 five_elem")
                
                try:
                    code_str = infer_code(five_elem)
                    print(f"  ✓ 已生成代码")
                except Exception as e:
                    print(f"  ✗ 代码生成失败: {str(e)}")
                    results.append({
                        'index': data.get('index', idx),
                        'question': question,
                        'five_elem': five_elem,
                        'code': None,
                        'output': None,
                        'error': f'Code generation failed: {str(e)}',
                        'ground_truth': data.get('answer', None)
                    })
                    continue
                
                out_log, err_log = test_code(code_str)
                
                if err_log and err_log.strip():
                    print(f"  ⚠️  代码执行有错误（但可能仍有输出）")
                else:
                    print(f"  ✓ 代码执行成功")
                
                if out_log:
                    print(f"  输出预览: {out_log[:100]}..." if len(out_log) > 100 else f"  输出: {out_log}")
                
                result = {
                    'index': data.get('index', idx),
                    'question': question,
                    'five_elem': five_elem,
                    'code': code_str,
                    'output': out_log,
                    'error': err_log if err_log and err_log.strip() else None,
                    'ground_truth': data.get('answer', None)
                }
                
                results.append(result)
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                fout.flush()                # 确保立即落盘
                print()
                
            except json.JSONDecodeError as e:
                print(f"  ✗ 第 {idx} 行 JSON 解析失败: {str(e)}\n")
                results.append({
                    'index': idx,
                    'question': '',
                    'error': f'JSON decode error: {str(e)}'
                })
            except Exception as e:
                print(f"  ✗ 第 {idx} 个问题处理失败: {str(e)}\n")
                results.append({
                    'index': data.get('index', idx) if 'data' in locals() else idx,
                    'question': data.get('question', '') if 'data' in locals() else '',
                    'error': str(e)
                })
    
    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\n{'='*60}")
        print(f"批量测试完成！")
        print(f"成功处理: {len([r for r in results if r.get('output') is not None])}/{len(results)}")
        print(f"结果已保存到: {output_path}")
        print(f"{'='*60}")
    
    return results


# 主程序
if __name__ == "__main__":
    # 批量测试配置
    test_file = "./data/testset/industryor.jsonl"
    output_file = "./inference/results/results_industryor.jsonl"
    
    # 检查测试文件是否存在
    test_file_path = Path(__file__).parent.parent / test_file.lstrip('./')
    if not test_file_path.exists():
        print(f"错误: 测试文件不存在: {test_file_path}")
        print("请确保文件路径正确")
        sys.exit(1)
    
    max_samples = 3  # None 表示全部
    batch_test(str(test_file_path), output_file, max_samples=max_samples)
    
    # 如果需要单个问题测试，可以取消下面的注释：
    # question = "你的问题..."
    # five_elem = infer_five_elem(question)
    # code_str = infer_code(five_elem)
    # out_log, err_log = test_code(code_str)
    # print(out_log)
