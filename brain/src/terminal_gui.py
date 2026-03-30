"""Simple curses interface for rating motor cortex output."""

from __future__ import annotations

import curses
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

import torch


class TerminalGUI(logging.Handler):
    """Interactive TUI displaying ``motor_cortex`` log messages."""

    def __init__(
        self,
        motor,
        buffer_seconds: float = 30.0,
        persist_path: str = "persistent/cli_feedback.log",
    ) -> None:
        super().__init__()
        self.motor = motor
        self.buffer_seconds = buffer_seconds
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.log: list[Tuple[float, str, int]] = []
        self.context: Deque[Tuple[float, torch.Tensor]] = deque()
        self.selected = -1
        self.rating = 0
        self.input_text = ""
        self.screen = None
        self.running = False

    # ``logging.Handler`` override
    def emit(self, record: logging.LogRecord) -> None:
        token = getattr(record, "token_id", -1)
        self.log.append((record.created, record.getMessage(), token))
        self.selected = len(self.log) - 1
        if self.screen:
            self.draw()

    def start(self) -> None:
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.running = True
        self.draw()

    def stop(self) -> None:
        if not self.screen:
            return
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()
        self.screen = None
        self.running = False

    def add_context(self, ctx: torch.Tensor) -> None:
        now = time.time()
        self.context.append((now, ctx.detach().cpu()))
        cutoff = now - self.buffer_seconds
        while self.context and self.context[0][0] < cutoff:
            self.context.popleft()

    def draw(self) -> None:
        if not self.screen:
            return
        self.screen.erase()
        h, w = self.screen.getmaxyx()
        left_w = w // 2
        self.screen.addstr(0, 0, "motor_cortex INFO".ljust(left_w))
        # draw log
        max_lines = h - 3
        start = max(0, len(self.log) - max_lines)
        for idx, (ts, msg, _) in enumerate(self.log[start:], start):
            y = idx - start + 1
            if y >= h - 2:
                break
            stamp = time.strftime("%H:%M:%S", time.localtime(ts)) + " "
            line = (stamp + msg)[: left_w - 1]
            if idx == self.selected:
                self.screen.attron(curses.A_REVERSE)
                self.screen.addstr(y, 0, line.ljust(left_w - 1))
                self.screen.attroff(curses.A_REVERSE)
            else:
                self.screen.addstr(y, 0, line.ljust(left_w - 1))

        # divider
        for y in range(h):
            self.screen.addch(y, left_w, curses.ACS_VLINE)

        # right pane rating
        x = left_w + 2
        self.screen.addstr(0, x, "Selected Output")
        for i, r in enumerate(range(-5, 6)):
            y = i + 2
            if y >= h - 2:
                break
            if r == self.rating:
                self.screen.attron(curses.A_REVERSE)
            self.screen.addstr(y, x, f"{r:+d}")
            if r == self.rating:
                self.screen.attroff(curses.A_REVERSE)

        self.screen.addstr(h - 2, x, self.input_text[: w - x - 1])
        self.screen.refresh()

    def handle_input(self) -> None:
        if not self.screen:
            return
        self.screen.timeout(0)
        ch = self.screen.getch()
        if ch == -1:
            return
        if ch == curses.KEY_UP:
            self.selected = max(0, self.selected - 1)
        elif ch == curses.KEY_DOWN:
            self.selected = min(len(self.log) - 1, self.selected + 1)
        elif ch == curses.KEY_RIGHT:
            if self.rating < 5:
                self.rating += 1
        elif ch == curses.KEY_LEFT:
            if self.rating > -5:
                self.rating -= 1
        elif ch in (curses.KEY_ENTER, 10, 13):
            self.apply_rating()
        elif ch in (curses.KEY_BACKSPACE, 127):
            self.input_text = self.input_text[:-1]
        elif ch == ord("\t"):
            self.submit_correction()
        elif ch == ord("q"):
            self.running = False
        elif 32 <= ch < 127:
            self.input_text += chr(ch)
        self.draw()

    def apply_rating(self) -> None:
        if not (0 <= self.selected < len(self.log)):
            return
        ts, msg, tok = self.log[self.selected]
        self.motor.reinforce_output(self.rating, tok)
        with self.persist_path.open("a") as f:
            f.write(f"{time.time()}\t{tok}\t{self.rating}\t{msg}\n")

    def submit_correction(self) -> None:
        text = self.input_text.strip()
        if not text or not self.context:
            self.input_text = ""
            return
        ctx = self.context[-1][1].to(self.motor.device)
        emb = self.motor.wernicke.encode([text]).mean(dim=1)
        self.motor._trainer.align(
            [self.motor.area.model.transformer, self.motor.damp_lora, self.motor.long_lora],
            ctx,
            emb.to(self.motor.device),
            lr_scale=1.0,
        )
        with self.persist_path.open("a") as f:
            f.write(f"{time.time()}\tTEACH\t{text}\n")
        self.input_text = ""

