"""Turning unexpected exceptions into a reason someone can act on."""
import asyncio

# Never swallowed: the user or the runtime is stopping the program.
_FATAL = (KeyboardInterrupt, SystemExit, asyncio.CancelledError)


def leaf_exceptions(error: BaseException) -> list[BaseException]:
    """The real exceptions inside (possibly nested) exception groups."""
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in leaf_exceptions(sub)]
    return [error]


def is_fatal(error: BaseException) -> bool:
    return any(isinstance(leaf, _FATAL) for leaf in leaf_exceptions(error))


def describe_error(error: BaseException) -> str:
    """The cause, not the wrapper.

    The executor's MCP tool session runs in a task group, so anything raised while it is open surfaces as
    "ExceptionGroup: unhandled errors in a TaskGroup", which says nothing about what failed.
    """
    leaves = leaf_exceptions(error)
    text = "; ".join(dict.fromkeys(f"{type(leaf).__name__}: {leaf}" for leaf in leaves))
    return text if not isinstance(error, BaseExceptionGroup) else f"{text} (raised inside a task group)"
