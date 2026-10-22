"""第1回: 素のLLMと会話するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 最初の質問を省略すると、講座固有の質問をする

ツールや用語集検索は使わないが、同じ実行中の会話履歴は保つ。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import chats
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(chat: chats.Chat, question: str) -> str:
    response = chat.send_message(question)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(f"[入力トークン数] {input_tokens if input_tokens is not None else '不明'}")
    print(f"[出力トークン数] {output_tokens if output_tokens is not None else '不明'}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保ちながら素のLLMに質問する")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="会話の最初の質問（省略時は講座固有の質問）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
        question = args.question
        while True:
            print(f"[質問] {question}\n")
            print(ask(chat, question))

            while True:
                try:
                    question = input("\n続けて質問してください（終了するには「終了」または exit）> ")
                except EOFError:
                    print()
                    return
                if question.strip().lower() in {"終了", "quit", "exit"}:
                    return
                if question.strip():
                    break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
