"""Action gating network."""

import torch
from torch import nn
from pathlib import Path
import time

from .utils.sentinel import SentinelLinear
from .subthalamic_nucleus import SubthalamicNucleus
from .caudate_nucleus import CaudateNucleus
from .putamen import Putamen
from .globus_pallidus import GlobusPallidus
from .nucleus_accumbens import NucleusAccumbens
from .substantia_nigra import SubstantiaNigra
from .premotor_cortex import PremotorCortex
from .inferior_frontal_gyrus import InferiorFrontalGyrus
from .supplementary_motor_area import SupplementaryMotorArea


class BasalGanglia(nn.Module):
    """Go/No-Go gating modulated by dopaminergic state."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        device: str = "cpu",
        axis: "HypothalamusPituitaryAxis | None" = None,
        prefrontal: "PrefrontalCortex | None" = None,
        premotor: "PremotorCortex | None" = None,
        ifg: "InferiorFrontalGyrus | None" = None,
        supplementary: "SupplementaryMotorArea | None" = None,
        stn: "SubthalamicNucleus | None" = None,
        persist_path: str | None = None,
        *,
        submodule_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.device = device
        self.axis = axis
        self.prefrontal = prefrontal
        self.premotor = premotor
        self.ifg = ifg
        self.supplementary = supplementary
        self.stn = stn
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        subdir = Path(submodule_dir) if submodule_dir else (self.persist_path.parent if self.persist_path else None)
        def p(name: str) -> str | None:
            return str(subdir / name) if subdir else None
        self.caudate = CaudateNucleus(input_dim, hidden_dim, device=device, persist_path=p("caudate_nucleus.pt"))
        self.putamen = Putamen(input_dim, hidden_dim, device=device, persist_path=p("putamen.pt"))
        self.pallidus = GlobusPallidus(input_dim, hidden_dim, device=device, persist_path=p("globus_pallidus.pt"))
        self.accumbens = NucleusAccumbens(input_dim, hidden_dim, device=device, persist_path=p("nucleus_accumbens.pt"))
        self.nigra = SubstantiaNigra(input_dim, hidden_dim, device=device, persist_path=p("substantia_nigra.pt"))
        self.prev_action: torch.Tensor | None = None
        self.last_output_time = 0.0
        self.feedback_pending = False
        self.last_rating = 0.0
        self.feedback_timeout = 2.0
        if self.persist_path and self.persist_path.exists():
            state = torch.load(self.persist_path, map_location=device)
            self.load_state_dict(state)

    def register_output(self) -> None:
        """Record that a motor action was produced."""

        self.last_output_time = time.time()
        self.feedback_pending = True
        self.last_rating = 0.0

    def register_feedback(self, rating: int) -> None:
        """Note user feedback on the last action."""

        self.feedback_pending = False
        self.last_rating = float(rating) / 5.0

    @torch.no_grad()
    def gate(self, embedding: torch.Tensor) -> bool:
        """Decide whether to produce a motor action for ``embedding``."""
        if self.feedback_pending and time.time() - self.last_output_time < self.feedback_timeout:
            return False

        prob = float(self.net(embedding.to(self.device)))
        prob *= self.caudate.evaluate(embedding)
        prob *= self.putamen.facilitate(embedding)
        prob *= 1.0 - self.pallidus.brake(embedding)
        prob += 0.3 * self.accumbens.reward_drive(embedding)
        prob += 0.2 * self.nigra.initiate(embedding)
        # Modulate gating probability using hormone levels if available
        if self.axis is not None:
            mod = (
                0.5 * float(self.axis.dopamine)
                + 0.2 * float(self.axis.norepinephrine)
                + 0.1 * float(self.axis.serotonin)
            )
            prob += mod
        if self.prefrontal is not None:
            pf = float(self.prefrontal(embedding.to(self.prefrontal.device)))
            prob *= pf
        else:
            pf = 0.5
        if self.premotor is not None:
            plan = self.premotor.prepare(embedding.to(self.premotor.device))
        else:
            plan = embedding
        if self.ifg is not None:
            inhibit_val = float(self.ifg.inhibit(embedding.to(self.ifg.device)))
        else:
            inhibit_val = 0.0
        if self.stn is not None:
            prob *= 1.0 - float(self.stn.inhibition(embedding))
        prob += 0.1 * self.last_rating
        prob = max(0.0, min(1.0, prob))
        threshold = 0.25
        if self.supplementary is not None:
            dopamine = float(self.axis.dopamine) if self.axis is not None else 0.5
            threshold = self.supplementary.compute_threshold(plan, pf, inhibit_val, dopamine)
        return prob > threshold

    @torch.no_grad()
    def approve_action(self, action: torch.Tensor) -> bool:
        """Return ``True`` if the proposed ``action`` should be executed."""

        action = action.to(self.device)
        prob = self.caudate.evaluate(action)
        prob *= 1.0 - self.pallidus.brake(action)
        prob += 0.2 * self.accumbens.reward_drive(action)
        if self.prev_action is not None:
            sim = torch.nn.functional.cosine_similarity(
                action.view(-1), self.prev_action.to(self.device).view(-1), dim=0
            ).item()
            # High similarity triggers braking and additional inhibition
            prob *= 1.0 - sim
            if self.stn is not None:
                prob *= 1.0 - sim * float(self.stn.inhibition(action))
        prob += 0.05 * self.last_rating
        prob = max(0.0, min(1.0, prob))
        approved = prob > 0.25
        if approved:
            self.prev_action = action.detach().cpu()
        return approved

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        self.caudate.save()
        self.putamen.save()
        self.pallidus.save()
        self.accumbens.save()
        self.nigra.save()
