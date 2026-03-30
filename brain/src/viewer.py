"""Utility for visualizing what brain sees using PyGame."""

from typing import Optional, Tuple
import numpy as np
import pygame


class Viewer:
    """Small window that displays the current frame with text and audio meter."""

    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        bar_height: int = 30,
        input_height: int = 20,
    ) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.bar_height = bar_height
        self.input_height = input_height
        self.screen = pygame.display.set_mode(
            (width, height + bar_height + input_height)
        )
        pygame.display.set_caption("brain Viewer")
        self.font = pygame.font.SysFont(None, 24)
        self.input_buffer = ""
        self.treat_rect = pygame.Rect(
            self.width - 70, self.height + 5, 65, self.bar_height - 10
        )

    def update(
        self,
        frame: np.ndarray,
        text: str = "",
        audio_level: float = 0.0,
        hormones: Optional[dict[str, float]] = None,
    ) -> None:
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        # Scale the surface so the entire image fits inside the viewer window
        if surface.get_width() != self.width or surface.get_height() != self.height:
            surface = pygame.transform.smoothscale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (0, self.height, self.width, self.bar_height + self.input_height),
        )
        clean_text = text.strip()
        if clean_text:
            try:
                txt_surf = self.font.render(clean_text, True, (255, 255, 255))
            except pygame.error:
                # fall back to a placeholder to avoid crashing on odd tokens
                txt_surf = self.font.render("[?]", True, (255, 255, 255))
            self.screen.blit(txt_surf, (5, self.height + 5))
        if hormones:
            items = [f"{k[0].upper()}:{float(v):.2f}" for k, v in hormones.items()]
            half = len(items) // 2 + len(items) % 2
            top = " ".join(items[:half])
            bottom = " ".join(items[half:])
            lvl_surf = self.font.render(top, True, (255, 255, 0))
            self.screen.blit(lvl_surf, (5, 5))
            if bottom:
                lvl_surf2 = self.font.render(bottom, True, (255, 255, 0))
                self.screen.blit(lvl_surf2, (5, 25))
        if self.input_buffer:
            input_surf = self.font.render(self.input_buffer, True, (255, 255, 255))
            self.screen.blit(input_surf, (5, self.height + self.bar_height + 5))
        meter_h = self.bar_height - 10
        level = int(max(0.0, min(1.0, audio_level)) * meter_h)
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (self.width - 15, self.height + meter_h - level + 5, 10, level),
        )
        pygame.draw.rect(self.screen, (255, 165, 0), self.treat_rect)
        label = self.font.render("Treat", True, (0, 0, 0))
        self.screen.blit(label, (self.treat_rect.x + 5, self.treat_rect.y))
        pygame.display.flip()
        pygame.event.pump()

    def close(self) -> None:
        pygame.quit()

    def poll_text_input(self) -> Tuple[Optional[str], bool]:
        """Return entered text and whether the treat button was clicked."""
        submitted: Optional[str] = None
        treat = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.treat_rect.collidepoint(event.pos):
                    treat = True
            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_RETURN:
                submitted = self.input_buffer
                self.input_buffer = ""
            elif event.key == pygame.K_BACKSPACE:
                self.input_buffer = self.input_buffer[:-1]
            else:
                if event.unicode and len(event.unicode) == 1 and ord(event.unicode) >= 32:
                    self.input_buffer += event.unicode

        return submitted, treat
