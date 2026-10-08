"""Command line interface:  python -m docqa <command>"""
import argparse
import sys

from docqa.config import Settings, load_settings
from docqa.evaluate import evaluate, load_questions
from docqa.pipeline import build_qa, build_store, download_demo, ingest_path
from docqa.qa import Answer


def print_answer(answer: Answer) -> None:
    print(f"\n{answer.text}")
    if answer.note:
        print(f"\n({answer.note})")
    for s in answer.sources:
        snippet = " ".join(s.text.split())
        print(f"\n[{s.number}] {s.label}  (similarity {s.score:.2f})\n    {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
    print()


def mode_line(settings: Settings) -> str:
    if settings.has_api_key:
        return f"Full AI mode: {settings.anthropic_model} writes answers from the retrieved passages."
    return "Search-only mode: no ANTHROPIC_API_KEY set. Add one to .env for written answers."


def cmd_ingest(args, settings: Settings) -> int:
    store = build_store(settings)
    for path in args.paths:
        try:
            total, new = ingest_path(store, settings, path)
        except (ValueError, OSError) as error:
            print(f"Skipped {path}: {error}")
            continue
        print(f"{path}: {total} chunks ({new} new)")
    print(f"Store now holds {store.count()} chunks.")
    return 0


def cmd_demo(args, settings: Settings) -> int:
    path = download_demo(settings)
    total, new = ingest_path(build_store(settings), settings, path)
    print(f"Demo paper ready: {total} chunks ({new} new). Try:  python -m docqa ask \"What is multi-head attention?\"")
    return 0


def cmd_ask(args, settings: Settings) -> int:
    print(mode_line(settings))
    print_answer(build_qa(settings).ask(args.question))
    return 0


def cmd_chat(args, settings: Settings) -> int:
    qa = build_qa(settings)
    print(mode_line(settings))
    print("Ask questions about your documents. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            print_answer(qa.ask(question))


def cmd_eval(args, settings: Settings) -> int:
    result = evaluate(build_store(settings), load_questions(args.questions), k=args.k)
    for d in result["details"]:
        print(f"  rank {d['rank'] or '-':>2}  {d['question']}")
    print(f"\n{result['questions']} questions | hit@{result['k']} = {result['hit_at_k']:.0%} | MRR = {result['mrr']:.2f}")
    return 0


def cmd_status(args, settings: Settings) -> int:
    store = build_store(settings)
    print(mode_line(settings))
    print(f"Embeddings: {settings.embeddings_backend} ({settings.embedding_model})")
    print(f"Chunks stored: {store.count()}")
    for source, chunks in store.sources().items():
        print(f"  {source}: {chunks} chunks")
    return 0


def cmd_reset(args, settings: Settings) -> int:
    build_store(settings).reset()
    print("Store cleared.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docqa", description="Ask questions about your documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="add PDF, .txt or .md files to the store")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_ingest)

    sub.add_parser("demo", help="download and ingest the 'Attention Is All You Need' paper").set_defaults(func=cmd_demo)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="interactive question loop").set_defaults(func=cmd_chat)

    p = sub.add_parser("eval", help="measure retrieval quality on a question set")
    p.add_argument("--questions", default="eval/attention_questions.json")
    p.add_argument("-k", type=int, default=4)
    p.set_defaults(func=cmd_eval)

    sub.add_parser("status", help="show mode and what is stored").set_defaults(func=cmd_status)
    sub.add_parser("reset", help="delete everything in the store").set_defaults(func=cmd_reset)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args, load_settings())


if __name__ == "__main__":
    sys.exit(main())
