"""Thread-safe in-memory state for interactive prompt reviews."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

MAX_REVIEW_TEXT_LENGTH = 100_000
_TOMBSTONE_TTL_SECONDS = 3600
_MAX_TOMBSTONES = 4096


class ReviewStateError(Exception):
    """Base class for expected review-state errors."""


class ReviewNotFound(ReviewStateError):
    pass


class ReviewAlreadyCompleted(ReviewStateError):
    pass


class InvalidReviewText(ReviewStateError):
    pass


class ReviewTextTooLong(ReviewStateError):
    pass


@dataclass(slots=True)
class PendingReview:
    review_id: str
    node_id: str
    client_id: str
    text: str
    created_at: float
    timeout_seconds: int
    deadline: float
    event: threading.Event = field(default_factory=threading.Event)
    result_text: str | None = None
    status: str = "pending"


class PromptReviewStore:
    """Owns pending reviews and bounded completion tombstones."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: dict[str, PendingReview] = {}
        self._completed: dict[str, float] = {}

    def _purge_completed_locked(self, now: float) -> None:
        cutoff = now - _TOMBSTONE_TTL_SECONDS
        expired = [review_id for review_id, finished_at in self._completed.items() if finished_at < cutoff]
        for review_id in expired:
            self._completed.pop(review_id, None)
        if len(self._completed) > _MAX_TOMBSTONES:
            oldest = sorted(self._completed, key=self._completed.get)
            for review_id in oldest[: len(self._completed) - _MAX_TOMBSTONES]:
                self._completed.pop(review_id, None)

    def create(self, node_id: str, client_id: str, text: str, timeout_seconds: int) -> PendingReview:
        now_monotonic = time.monotonic()
        with self._lock:
            self._purge_completed_locked(now_monotonic)
            review_id = uuid.uuid4().hex
            while review_id in self._pending or review_id in self._completed:
                review_id = uuid.uuid4().hex
            pending = PendingReview(
                review_id=review_id,
                node_id=str(node_id),
                client_id=str(client_id),
                text=str(text),
                created_at=time.time(),
                timeout_seconds=int(timeout_seconds),
                deadline=now_monotonic + int(timeout_seconds),
            )
            self._pending[review_id] = pending
            return pending

    def submit(self, review_id: str, text: str) -> PendingReview:
        if not isinstance(text, str) or not text.strip():
            raise InvalidReviewText("Reviewed prompt must not be empty.")
        if len(text) > MAX_REVIEW_TEXT_LENGTH:
            raise ReviewTextTooLong("Reviewed prompt exceeds the maximum length.")
        now = time.monotonic()
        with self._lock:
            self._purge_completed_locked(now)
            pending = self._pending.get(review_id)
            if pending is None:
                if review_id in self._completed:
                    raise ReviewAlreadyCompleted("Review has already been completed.")
                raise ReviewNotFound("Review does not exist.")
            if pending.status != "pending":
                raise ReviewAlreadyCompleted("Review has already been completed.")
            pending.result_text = text
            pending.status = "approved"
            pending.event.set()
            return pending

    def mark_terminal(self, review_id: str, status: str) -> PendingReview | None:
        with self._lock:
            pending = self._pending.get(review_id)
            if pending is None or pending.status != "pending":
                return pending
            pending.status = status
            pending.event.set()
            return pending

    def get(self, review_id: str) -> PendingReview | None:
        with self._lock:
            return self._pending.get(review_id)

    def pending_for_client(self, client_id: str) -> list[dict]:
        with self._lock:
            return [
                {
                    "review_id": pending.review_id,
                    "node_id": pending.node_id,
                    "text": pending.text,
                    "timeout_seconds": pending.timeout_seconds,
                    "created_at": pending.created_at,
                }
                for pending in self._pending.values()
                if pending.client_id == client_id and pending.status == "pending"
            ]

    def cleanup(self, review_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            pending = self._pending.pop(review_id, None)
            if pending is not None:
                self._completed[review_id] = now
            self._purge_completed_locked(now)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._completed.clear()


PROMPT_REVIEW_STORE = PromptReviewStore()
