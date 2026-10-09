"""Structured-output calls that survive a cut-off or malformed reply."""
import json

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

# What a truncated or malformed structured reply raises. A reply cut off mid-JSON gets partially repaired by the
# parser and then fails schema validation ("N validation errors ... Field required").
PARSE_ERRORS = (OutputParserException, ValidationError, json.JSONDecodeError)

# Output limit for structured calls. Thinking counts against the limit, so on a hard task a long think can leave too
# little room for the JSON; this doubles the model default.
STRUCTURED_MAX_TOKENS = 32_000
# The Anthropic SDK refuses non-streaming requests above ~21K max_tokens (they could outlive the HTTP timeout).
_NONSTREAMING_MAX_TOKENS = 21_000

RETRY_REMINDER = (
    "Your previous reply could not be used: it was cut off or was not valid JSON for the required schema. "
    "Reply again with the complete JSON object. Keep every string short and every list brief so it fits."
)


class StructuredOutputError(Exception):
    """The model did not produce a parseable reply, even after one retry."""


def call_structured(llm, schema, messages: list, *, run_name: str, metadata: dict | None = None,
                    max_tokens: int = STRUCTURED_MAX_TOKENS):
    """Returns (parsed result, retried). Retries once with a reminder; raises StructuredOutputError after that."""
    runnable = llm.with_structured_output(schema, method="json_schema")
    kwargs = {"max_tokens": max_tokens}
    if max_tokens > _NONSTREAMING_MAX_TOKENS:
        kwargs["stream"] = True

    def ask(msgs: list, name: str):
        result = runnable.invoke(msgs, config={"run_name": name, "metadata": metadata or {}}, **kwargs)
        if result is None:
            raise OutputParserException("the model returned no structured output")
        return result

    try:
        return ask(messages, run_name), False
    except PARSE_ERRORS:
        pass
    try:
        return ask([*messages, ("human", RETRY_REMINDER)], f"{run_name} (retry)"), True
    except PARSE_ERRORS as e:
        raise StructuredOutputError(f"{run_name}: reply unparseable twice: {str(e)[:300]}") from e
