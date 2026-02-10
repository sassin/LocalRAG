# memory.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple
from collections import deque


@dataclass
class SessionState:
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # Keep the last N turns only (memory-light)
    turns: Deque[Tuple[str, str]] = field(default_factory=lambda: deque(maxlen=8))  # (user, assistant)
    # A tiny rolling "summary" (heuristic, no LLM call)
    summary: str = ""
    # Recently used sources (paper.pdf p.X)
    recent_sources: Deque[str] = field(default_factory=lambda: deque(maxlen=8))

    def add_turn(self, user: str, assistant: str):
        self.last_seen = time.time()
        self.turns.append((user, assistant))

    def add_sources(self, sources: List[str]):
        self.last_seen = time.time()
        for s in sources:
            if s and s not in self.recent_sources:
                self.recent_sources.appendleft(s)

    def build_context_block(self) -> str:
        """
        Small, stable memory block: summary + last 2 turns + recent sources.
        Keeps the prompt compact and avoids context blow-up.
        """
        parts: List[str] = []

        if self.summary.strip():
            parts.append(f"Conversation summary:\n{self.summary.strip()}")

        # Last 2 turns for immediate follow-up coherence
        if self.turns:
            tail = list(self.turns)[-2:]
            lines = []
            for u, a in tail:
                lines.append(f"User: {u}")
                lines.append(f"Assistant: {a}")
            parts.append("Recent turns:\n" + "\n".join(lines))

        if self.recent_sources:
            parts.append("Recently referenced sources:\n" + "\n".join(list(self.recent_sources)[:5]))

        return "\n\n".join(parts).strip()

    def update_summary_heuristic(self):
        """
        Heuristic: keep it short and topic-focused.
        No extra LLM calls.
        """
        # Extract key user intents from recent user messages (very lightweight)
        user_msgs = [u for u, _ in self.turns][-4:]
        if not user_msgs:
            return

        # Keep ~500 chars summary max
        combined = " | ".join(m.strip().replace("\n", " ") for m in user_msgs if m.strip())
        combined = combined[:500]

        # Stable phrasing helps the model
        self.summary = combined


class InMemoryChatStore:
    def __init__(self, ttl_seconds: int = 6 * 60 * 60):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        now = time.time()
        self._gc(now)
        st = self._sessions.get(session_id)
        if not st:
            st = SessionState()
            self._sessions[session_id] = st
        st.last_seen = now
        return st

    def _gc(self, now: float):
        dead = []
        for sid, st in self._sessions.items():
            if now - st.last_seen > self.ttl_seconds:
                dead.append(sid)
        for sid in dead:
            self._sessions.pop(sid, None)
