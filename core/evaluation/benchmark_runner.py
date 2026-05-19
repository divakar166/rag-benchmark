"""
Benchmark Runner — Phase 3 CLI.

Run from the project root:
    uv run python -m core.evaluation.benchmark_runner --strategies naive
    uv run python -m core.evaluation.benchmark_runner --strategies naive semantic hybrid
    uv run python -m core.evaluation.benchmark_runner --top-k 3
"""
import argparse
import asyncio
from pathlib import Path

from loguru import logger


def _print_results_table(summary: dict) -> None:
    """Print a formatted comparison table to stdout."""
    strategies = summary["strategies"]

    print()
    print("RAG STRATEGY BENCHMARK RESULTS")
    print()

    col_w = 16
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    header = (
        f"{'Strategy':<16}"
        f" {'Faithfulness':>{col_w}}"
        f" {'Ans.Relevancy':>{col_w}}"
        f" {'Ctx.Precision':>{col_w}}"
        f" {'Ctx.Recall':>{col_w}}"
        f" {'Avg Latency':>{col_w}}"
        f" {'Avg Tokens':>{col_w}}"
    )
    print(header)
    print("-" * 100)

    def fmt(v) -> str:
        """Format a score value — shows N/A cleanly for None."""
        return f"{v:.4f}" if v is not None else "   N/A  "

    # Sort by faithfulness descending (None treated as 0)
    sorted_strategies = sorted(
        strategies.items(),
        key=lambda x: x[1]["ragas"].get("faithfulness") or 0,
        reverse=True,
    )

    for strategy, data in sorted_strategies:
        ragas = data["ragas"]
        line = (
            f"{strategy:<16}"
            f" {fmt(ragas.get('faithfulness')):>{col_w}}"
            f" {fmt(ragas.get('answer_relevancy')):>{col_w}}"
            f" {fmt(ragas.get('context_precision')):>{col_w}}"
            f" {fmt(ragas.get('context_recall')):>{col_w}}"
            f" {data['latency_avg_ms']:>{col_w - 2}.0f}ms"
            f" {data['tokens_avg']:>{col_w}.0f}"
        )
        print(line)


    print("\nWINNER PER METRIC:")
    for metric in metrics:
        # Filter to strategies that actually have a score for this metric
        scored = [
            (name, data)
            for name, data in strategies.items()
            if data["ragas"].get(metric) is not None
        ]
        if not scored:
            print(f"  {metric:<22} → {'N/A (no scores yet)'}")
            continue

        best_name, best_data = max(scored, key=lambda x: x[1]["ragas"][metric])
        score = best_data["ragas"][metric]
        print(f"  {metric:<22} → {best_name:<16} ({score:.4f})")

    fastest = min(strategies.items(), key=lambda x: x[1]["latency_avg_ms"])
    print(
        f"  {'fastest':<22} → {fastest[0]:<16} "
        f"({fastest[1]['latency_avg_ms']:.0f}ms avg)"
    )
    print()


async def _main(args: argparse.Namespace) -> None:
    from core.evaluation.ragas_evaluator import run_evaluation

    question_set = Path(args.question_set)
    if not question_set.exists():
        logger.error(f"Question set not found: {question_set}")
        return

    strategies = args.strategies if args.strategies else None

    logger.info("Starting RAG benchmark evaluation...")
    logger.info(f"  Question set : {question_set}")
    logger.info(f"  Strategies   : {strategies or 'all'}")
    logger.info(f"  Top-K        : {args.top_k}")
    logger.info(f"  Output dir   : {args.output_dir}")

    result = await run_evaluation(
        question_set_path=question_set,
        strategies=strategies,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )

    _print_results_table(result["summary"])

    print("Full results saved to:")
    print(f"  CSV  : {result['csv_path']}")
    print(f"  JSON : {result['json_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG strategy benchmark evaluation using RAGAS"
    )
    parser.add_argument(
        "--question-set",
        default="core/evaluation/question_set.json",
        help="Path to question_set.json",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["naive", "semantic", "hierarchical", "hybrid", "hyde"],
        help="Strategies to evaluate (default: all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="core/evaluation/results",
        help="Directory to save results (default: core/evaluation/results)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()