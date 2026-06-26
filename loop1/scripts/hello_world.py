"""Phase 0 sanity check — one LLM call, one log event."""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loop1.config import config
from loop1.llm import call_llm
from loop1.logging_utils import log_event


def main() -> None:
    session_id = str(uuid.uuid4())
    model = config["models"]["doctor"]

    print(f"Model : {model}")
    print("Calling LLM...")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with exactly: hello world"},
    ]

    response = call_llm(model=model, messages=messages)
    print(f"Response: {response}")

    log_event(session_id, "hello_world", {"model": model, "response": response})
    print(f"Logged  : logs/sessions/session_{session_id}.jsonl")
    print("Phase 0 sanity check passed.")


if __name__ == "__main__":
    main()
