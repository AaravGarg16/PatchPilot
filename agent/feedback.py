# Keeps prompts bounded when tool output is noisy; pytest puts its failure summary at the end.
MAX_OUTPUT_CHARS = 6000


def tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return "[earlier output truncated]\n" + text[-limit:]
