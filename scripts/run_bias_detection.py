#!/usr/bin/env python3
"""
一键运行偏见检测测试

使用说明:
    python scripts/run_bias_detection.py

此脚本会:
1. 使用偏见检测数据集运行 inference
2. 自动分析结果
3. 生成报告
"""

import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主函数"""
    print("=" * 80)
    print("LLM偏见检测测试")
    print("=" * 80)

    # 检查测试数据集是否存在
    test_file = project_root / "data/testset/bias_detection_supplier.jsonl"
    if not test_file.exists():
        print(f"❌ 测试数据集不存在: {test_file}")
        print("   请先创建偏见检测数据集")
        return

    print(f"\n✓ 使用测试数据集: {test_file}")

    # 准备输出目录
    results_dir = project_root / "inference" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"bias_detection_{timestamp}.jsonl"

    print(f"✓ 结果将保存到: {output_file}")

    # 运行 inference
    print("\n" + "-" * 80)
    print("第1步: 运行模型推理...")
    print("-" * 80)

    try:
        from inference.client_batch_server import batch_from_jsonl

        results = batch_from_jsonl(
            str(test_file), str(output_file), max_samples=None
        )
        print(f"\n✓ 推理完成，共处理 {len(results)} 个问题")
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        print("\n请确保:")
        print("  1. 模型服务已启动 (model_server.py)")
        print("  2. 服务地址和端口配置正确")
        return

    # 分析结果
    print("\n" + "-" * 80)
    print("第2步: 分析偏见检测结果...")
    print("-" * 80)

    try:
        from scripts.analyze_bias_simple import SimpleBiasAnalyzer

        analyzer = SimpleBiasAnalyzer()
        analyzer.load_results(str(output_file))
        analyzer.print_report()
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        print(f"\n请手动运行分析:")
        print(f"  python scripts/analyze_bias_simple.py --results {output_file}")
        return

    print("\n" + "=" * 80)
    print("偏见检测完成！")
    print("=" * 80)
    print(f"\n详细结果已保存到: {output_file}")
    print(f"\n可以使用以下命令重新查看分析:")
    print(f"  python scripts/analyze_bias_simple.py --results {output_file}")


if __name__ == "__main__":
    main()
