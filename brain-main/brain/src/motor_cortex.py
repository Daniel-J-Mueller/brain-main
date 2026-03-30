"""Text motor output using the back half of GPT-2."""

import torch
from torch import nn
from pathlib import Path
import time
from collections import deque
from typing import Callable

from .language_areas.brocas_area import BrocasArea
from .language_areas.wernickes_area import WernickesArea
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras
from .utils.sentinel import SentinelLinear
from .utils.curiosity import CuriosityTracker
from .hypothalamus_pituitary_axis import HypothalamusPituitaryAxis
from .basal_ganglia import BasalGanglia
from .inferior_frontal_gyrus import InferiorFrontalGyrus
from .trainer import Trainer
from .utils.logger import get_logger


class MotorCortex(nn.Module):
    """Generates text from context embeddings and prints it."""

    def __init__(
        self,
        model_dir: str,
        wernicke: WernickesArea,
        device: str = "cpu",
        axis: HypothalamusPituitaryAxis | None = None,
        persist_path: str | None = None,
        num_candidates: int = 1,
        feedback_buffer: float = 30.0,
        basal: "BasalGanglia | None" = None,
        ifg: "InferiorFrontalGyrus | None" = None,
    ) -> None:
        super().__init__()
        self.logger = get_logger("motor_cortex")
        self.area = BrocasArea(model_dir, device=device)
        self.wernicke = wernicke
        self.axis = axis
        self.device = device
        self.vision_to_text = SentinelLinear(128, self.area.model.config.n_embd).to(device)
        self.damp_lora = FatigueLoRA(
            self.area.model.config.n_embd,
            self.area.model.config.n_embd,
            device=device,
        )
        self.long_lora = LongTermLoRA(
            self.area.model.config.n_embd,
            self.area.model.config.n_embd,
            device=device,
        )
        self.curiosity = CuriosityTracker()
        self._trainer = Trainer()
        self.feedback_buffer = float(feedback_buffer)
        self.basal = basal
        self.ifg = ifg
        self._recent = deque()  # (timestamp, token_id, context, token_emb)
        if persist_path and Path(persist_path).exists():
            state = torch.load(persist_path, map_location=device)
            self.vision_to_text.load_state_dict(state.get("vision_to_text", {}), strict=False)
            self.damp_lora.load_state_dict(state.get("damp_lora", {}), strict=False)
            self.long_lora.load_state_dict(state.get("long_lora", {}), strict=False)
            if "curiosity" in state:
                self.curiosity.load_state_dict(state["curiosity"])
        self.persist_path = persist_path
        self.num_candidates = max(1, int(num_candidates))
        self.history: list[int] = []
        self.history_size = 50
        self.repetition_penalty = 1.2

    def _trim_recent(self) -> None:
        cutoff = time.time() - self.feedback_buffer
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    @torch.no_grad()
    def reinforce_output(self, rating: int, token_id: int) -> None:
        """Strengthen or weaken the association for ``token_id``."""
        if self.basal is not None:
            self.basal.register_feedback(rating)
        if self.axis is not None:
            self.axis.update_valence(float(rating) / 5.0)
        if not self._recent or rating == 0:
            return
        entry = None
        for ts, tid, ctx, emb in reversed(self._recent):
            if tid == token_id:
                entry = (ctx, emb)
                break
        if entry is None:
            return
        ctx, emb = entry
        scale = abs(float(rating)) / 5.0
        target = emb.to(self.device) if rating > 0 else torch.zeros_like(emb)
        self._trainer.align(
            [self.area.model.transformer, self.damp_lora, self.long_lora],
            ctx.to(self.device),
            target,
            lr_scale=scale,
        )
        if self.ifg is not None:
            self.ifg.reinforce(rating)

    def modules(self):
        """Yield child modules for initialization."""
        for m in (self.vision_to_text, self.damp_lora, self.long_lora):
            yield m

    # enable loading via :func:`maybe_initialize`
    def load_state_dict(self, state: dict, strict: bool = False):
        self.vision_to_text.load_state_dict(state.get("vision_to_text", {}), strict=strict)
        self.damp_lora.load_state_dict(state.get("damp_lora", {}), strict=strict)
        self.long_lora.load_state_dict(state.get("long_lora", {}), strict=strict)
        if "curiosity" in state:
            self.curiosity.load_state_dict(state["curiosity"])

    def state_dict(self):
        return {
            "vision_to_text": self.vision_to_text.state_dict(),
            "damp_lora": self.damp_lora.state_dict(),
            "long_lora": self.long_lora.state_dict(),
            "curiosity": self.curiosity.state_dict(),
        }
        
    def save(self, path: str | None = None) -> None:
        """Save adapter parameters for later reloading."""
        target = path or self.persist_path
        if not target:
            return
        torch.save(
            {
                "vision_to_text": self.vision_to_text.state_dict(),
                "damp_lora": self.damp_lora.state_dict(),
                "long_lora": self.long_lora.state_dict(),
                "curiosity": self.curiosity.state_dict(),
            },
            target,
        )
        save_loras(self, target)

    @torch.no_grad()
    def act(
        self,
        hidden: torch.Tensor,
        temperature: float | None = None,
        num_candidates: int | None = None,
        valence_fn: "Callable[[torch.Tensor], torch.Tensor] | None" = None,
    ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, list[str]]:
        """Return the token whose embedding best matches ``hidden``.

        When the precomputed token table is available, similarity against the
        entire vocabulary determines the most appropriate token. Otherwise the
        method falls back to sampling candidate tokens from Broca's area and
        selecting the closest one via :class:`WernickesArea`. The embeddings for
        the ``num_candidates`` closest tokens are returned for training so that
        alignment can reinforce context associations.

        Parameters
        ----------
        hidden:
            Context embedding from the DMN.
        temperature:
            Optional sampling temperature.  ``None`` uses the current
            norepinephrine level.
        num_candidates:
            Number of speculative tokens to generate.  When ``None`` the
            ``MotorCortex`` instance's ``num_candidates`` value is used.
        valence_fn:
            Optional callable that receives the candidate embeddings and
            returns predicted valence scores.  The scores are added to the
            similarity measure when selecting the final token.
        Returns
        -------
        tuple
            ``(text, emb, feedback_emb, all_embs, index, texts)`` where ``text``
            is the emitted string, ``emb`` is its embedding, ``feedback_emb`` is
            the embedding of the runner up token used for feedback,
            ``all_embs`` are embeddings for every candidate, ``index`` is the
            chosen candidate's position within ``all_embs`` and ``texts``
            contains every speculative token.
        """

        temp = temperature
        if temp is None:
            if self.axis is not None:
                temp = 1.0 + float(self.axis.norepinephrine)
            else:
                temp = 1.0

        n = num_candidates if num_candidates is not None else self.num_candidates
        if self.wernicke.token_table is not None:
            n = max(2, n)

        # Apply adaptive dampening to the hidden state
        hidden = hidden.to(self.device)
        hidden = hidden + self.damp_lora(hidden) + self.long_lora(hidden)
        hidden = self.curiosity.transform(hidden)

        if self.wernicke.token_table is not None:
            # Directly pick the most similar tokens from the precomputed table
            table = self.wernicke.token_table
            context_vec = hidden.to(table.device)
            if context_vec.dim() == 3:
                context_vec = context_vec.mean(dim=1)
            context_vec = context_vec.squeeze(0)
            sims = torch.nn.functional.cosine_similarity(table, context_vec, dim=1)
            sims = sims / max(temp, 1e-5)
            if self.history:
                hist_ids = torch.tensor(self.history, dtype=torch.long, device=sims.device)
                sims[hist_ids] *= 1.0 / self.repetition_penalty
            curiosity_bonus = torch.tensor([
                self.curiosity.bonus(i) for i in range(table.size(0))
            ], device=sims.device)
            sims = sims * (1.0 + curiosity_bonus)
            topk = torch.topk(sims, k=n)
            ids = topk.indices.tolist()
            texts = [self.wernicke.tokenizer.decode([i], skip_special_tokens=True) for i in ids]
            enc = table[ids].unsqueeze(1)
            scores = topk.values
            if valence_fn is not None:
                val = valence_fn(enc)
                scores = scores + val
            order = torch.argsort(scores, descending=True)
            best_idx = int(order[0].item())
            fb_idx = int(order[1].item()) if order.numel() > 1 else best_idx
            best_text = texts[best_idx]
            fb_text = texts[fb_idx]
        else:
            # Fall back to sampling via Broca's area when no table is available
            candidates = list(
                self.area.decode(
                    hidden.to(self.device),
                    temperature=temp,
                    num_samples=n,
                    history=self.history,
                    repetition_penalty=self.repetition_penalty,
                )
            )
            texts = [t for t, _, _ in candidates]
            ids = [tid for _, _, tid in candidates]
            enc = self.wernicke.encode(texts)
            enc_means = enc.mean(dim=1)
            context_vec = hidden.to(enc_means.device)
            if context_vec.dim() == 3:
                context_vec = context_vec.mean(dim=1)
            sims = torch.nn.functional.cosine_similarity(enc_means, context_vec.squeeze(0), dim=1)
            curiosity_bonus = torch.tensor([
                self.curiosity.bonus(tid) for tid in ids
            ], device=sims.device)
            sims = sims * (1.0 + curiosity_bonus)
            scores = sims
            if valence_fn is not None:
                val = valence_fn(enc)
                scores = scores + val
            order = torch.argsort(scores, descending=True)
            best_idx = int(order[0].item())
            fb_idx = int(order[1].item()) if order.numel() > 1 else best_idx
            best_text = texts[best_idx]
            fb_text = texts[fb_idx]

        # update history of produced tokens
        if ids:
            tok_id = ids[best_idx]
            self.history.append(tok_id)
            if len(self.history) > self.history_size:
                self.history.pop(0)
            self.curiosity.update(tok_id)
            fb_id = ids[fb_idx]
        else:
            tok_id = -1
            fb_id = -1

        self.logger.info(best_text, extra={"token_id": tok_id})
        chosen_emb = enc[best_idx : best_idx + 1]
        fb_emb = enc[fb_idx : fb_idx + 1]
        self._recent.append(
            (
                time.time(),
                tok_id,
                hidden.detach().cpu(),
                chosen_emb.detach().cpu(),
            )
        )
        self._trim_recent()
        if self.ifg is not None:
            self.ifg.record_output(hidden)
        return best_text, chosen_emb, fb_emb, enc, best_idx, texts

    @torch.no_grad()
    def learn_from_feedback(
        self,
        vision_feat: torch.Tensor,
        audio_emb: torch.Tensor,
        motor_embs: torch.Tensor,
        trainer: Trainer,
    ) -> None:
        """Align motor output with visual and auditory context.

        ``motor_embs`` contains the embeddings for *all* speculative tokens
        generated during :meth:`act`. Each candidate token contributes to
        adaptation by being aligned separately against the visual and auditory
        cues. This ensures learning incorporates every possibility even though
        only one token is ultimately emitted.
        """
        vision_target = self.vision_to_text(vision_feat.to(self.device))
        for emb in motor_embs:
            # ``emb`` has shape ``(seq_len, hidden)``. Preserve token-level
            # information by aligning each token separately.
            for tok in emb:
                tok = tok.unsqueeze(0)
                trainer.align(
                    [self.area.model.transformer, self.vision_to_text],
                    vision_target,
                    tok,
                )
                trainer.align(
                    [self.area.model.transformer],
                    audio_emb.to(self.device),
                    tok,
                )
                trainer.align(
                    [self.damp_lora],
                    torch.zeros_like(tok),
                    tok,
                )
                trainer.align(
                    [self.long_lora],
                    torch.zeros_like(tok),
                    tok,
                )
