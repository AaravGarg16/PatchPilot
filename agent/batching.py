"""Send chat-model calls through Anthropic's Message Batches API (half the price, asynchronous).

Each call still looks synchronous to the agent: it waits until its batch has ended. A background thread groups
calls arriving within a short window into one batch, so concurrent eval runs share batches. The cost is latency:
every agent step waits for a batch to finish, which usually takes minutes and can take much longer.
"""
import asyncio
import itertools
import threading
import time
from concurrent.futures import Future
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr

# Parameters messages.create accepts but a batch request does not.
_NOT_BATCHABLE = {"stream"}


class BatchError(RuntimeError):
    pass


class BatchScheduler:
    def __init__(self, client, *, window_seconds: float = 5.0, max_batch: int = 100, poll_seconds: float = 30.0,
                 max_wait_seconds: float = 24 * 3600):
        self._client = client
        self.window_seconds = window_seconds
        self.max_batch = max_batch
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self._ids = itertools.count()
        self._cond = threading.Condition()
        self._pending: list[tuple[str, dict, Future, float]] = []
        self._in_flight: dict[str, tuple[dict[str, Future], float]] = {}
        self._thread: threading.Thread | None = None
        self.batches_submitted = 0

    def submit(self, params: dict) -> Future:
        future: Future = Future()
        with self._cond:
            self._pending.append((f"req-{next(self._ids)}", params, future, time.monotonic()))
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="anthropic-batches", daemon=True)
                self._thread.start()
            self._cond.notify()
        return future

    def _loop(self) -> None:
        last_poll = 0.0
        while True:
            with self._cond:
                while not self._pending and not self._in_flight:
                    self._cond.wait()
                now = time.monotonic()
                ready = self._pending and (
                    len(self._pending) >= self.max_batch or now - self._pending[0][3] >= self.window_seconds
                )
                to_send = self._pending[: self.max_batch] if ready else []
                if to_send:
                    self._pending = self._pending[len(to_send):]
            if to_send:
                self._send(to_send)
            if self._in_flight and time.monotonic() - last_poll >= self.poll_seconds:
                last_poll = time.monotonic()
                self._poll()
            with self._cond:
                self._cond.wait(timeout=min(self.window_seconds, self.poll_seconds) / 2)

    def _send(self, requests: list[tuple[str, dict, Future, float]]) -> None:
        try:
            batch = self._client.messages.batches.create(
                requests=[{"custom_id": cid, "params": params} for cid, params, _, _ in requests]
            )
        except Exception as e:  # noqa: BLE001 - every caller in the failed batch gets the error
            for _, _, future, _ in requests:
                future.set_exception(BatchError(f"could not create batch: {type(e).__name__}: {e}"))
            return
        self.batches_submitted += 1
        self._in_flight[batch.id] = ({cid: future for cid, _, future, _ in requests}, time.monotonic())

    def _poll(self) -> None:
        for batch_id, (futures, started) in list(self._in_flight.items()):
            try:
                status = self._client.messages.batches.retrieve(batch_id).processing_status
                if status != "ended":
                    if time.monotonic() - started > self.max_wait_seconds:
                        self._client.messages.batches.cancel(batch_id)
                        self._fail(futures, f"batch {batch_id} still running after {self.max_wait_seconds:.0f}s")
                        del self._in_flight[batch_id]
                    continue
                for entry in self._client.messages.batches.results(batch_id):
                    future = futures.pop(entry.custom_id, None)
                    if future is None:
                        continue
                    if entry.result.type == "succeeded":
                        future.set_result(entry.result.message)
                    else:
                        detail = getattr(entry.result, "error", None)
                        future.set_exception(BatchError(f"batch request {entry.result.type}: {detail}"))
                self._fail(futures, f"batch {batch_id} ended without a result for this request")
                del self._in_flight[batch_id]
            except Exception as e:  # noqa: BLE001 - a polling error is retried on the next poll
                if time.monotonic() - started > self.max_wait_seconds:
                    self._fail(futures, f"batch {batch_id} could not be read: {type(e).__name__}: {e}")
                    del self._in_flight[batch_id]

    @staticmethod
    def _fail(futures: dict[str, Future], reason: str) -> None:
        for future in futures.values():
            if not future.done():
                future.set_exception(BatchError(reason))


class BatchChatAnthropic(ChatAnthropic):
    """ChatAnthropic whose requests go through the Batch API. Tools, structured output and usage work unchanged:
    the request payload and the response parsing are ChatAnthropic's own."""

    batch_window_seconds: float = 5.0
    batch_poll_seconds: float = 30.0
    batch_max_wait_seconds: float = 24 * 3600
    _scheduler: BatchScheduler | None = PrivateAttr(default=None)

    @property
    def scheduler(self) -> BatchScheduler:
        if self._scheduler is None:
            self._scheduler = BatchScheduler(
                self._client, window_seconds=self.batch_window_seconds, poll_seconds=self.batch_poll_seconds,
                max_wait_seconds=self.batch_max_wait_seconds,
            )
        return self._scheduler

    def _should_stream(self, **kwargs) -> bool:
        # A caller asking for streaming (large max_tokens) still goes through a batch, which has no such limit.
        return False

    def _batch_params(self, messages, stop, **kwargs) -> dict:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        if "betas" in payload:
            raise BatchError("beta features are not wired into batch mode")
        return {k: v for k, v in payload.items() if k not in _NOT_BATCHABLE}

    def _result(self, message: Any, **kwargs) -> ChatResult:
        result = self._format_output(message, **kwargs)
        for generation in result.generations:
            # Marks the call for the usage tracker, which prices batch tokens at half.
            generation.message.response_metadata["batch"] = True
        return result

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.scheduler.submit(self._batch_params(messages, stop, **kwargs)).result()
        return self._result(message, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        future = self.scheduler.submit(self._batch_params(messages, stop, **kwargs))
        message = await asyncio.wrap_future(future)
        return self._result(message, **kwargs)
