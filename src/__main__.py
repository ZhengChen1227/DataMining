import argparse
from .data import prepare
from .experiment import run
from .summary import summarize


def main():
    parser = argparse.ArgumentParser(description="ABLP 课程项目：数据、实验与汇总")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="下载两个小型 SNAP 数据集")
    commands.add_parser("smoke", help="合成图端到端冒烟实验")
    experiment = commands.add_parser("run", help="运行或继续实验矩阵")
    experiment.add_argument("--config", default="configs/stage1.yaml")
    summary = commands.add_parser("summarize", help="从实际结果生成表格和图")
    summary.add_argument("--results", default="results/stage1")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "smoke":
        return run("configs/smoke.yaml")
    elif args.command == "run":
        return run(args.config)
    else:
        summarize(args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
