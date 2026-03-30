# Revised Neurosymbolic Plan

This document outlines how we will evolve brain from the current single-model approach into a set of cooperating region models. Each region will roughly match the size and function of its biological counterpart as described in `brain/docs/human_brain_components_reference.txt`. Persistent checkpoints currently located in `brain/persistent/` (e.g. `motor_cortex_adapters.pt`, `motor_insula.pt`, `wernicke_adapter.pt`) are treated as temporary bootstrap weights.

## 1. Core Principles

- Keep each brain region independent with its own model file.
- Use the base models in `brain/models/` as seed weights for specialised LoRA adapters.
- When possible, new models should share a common format (PyTorch `.pt` or NumPy `.npy`) to avoid additional loaders.
- Reserve unusual sentinel values for untrained connections so that a region can skip processing them until reinforced.
- Align neighbouring regions on the same GPU to minimise cross-device copies.

## 2. Region Allocation

We divide the system into smaller modules. For each area we define the approximate parameter budget relative to the 16 billion cortical neurons.

| Region                      | Model Type/Size (approx.) | GPU |
|-----------------------------|---------------------------|-----|
| Sensory Cortex (visual/auditory) | Small CNN/Conv1D (~5M) | 0 |
| Thalamus & DMN              | Transformer (~15M)        | 1 |
| Hippocampus                 | FAISS index + MLP (~10M)  | 1 |
| Basal Ganglia               | Gating MLP (~5M)          | 2 |
| Cerebellum                  | MLP (~3M)                 | 2 |
| Prefrontal Cortex & OFC     | Transformer (~10M)        | 2 |
| Motor & Insular Cortex      | GPT‑2 half + projections (~60M) | 3 |

This layout keeps sensory preprocessing together, while higher order decision and motor areas share GPUs to reduce latency between them. The run scripts already enforce these assignments using ``CUDA_VISIBLE_DEVICES``.

## 3. Bootstrapping Status

Core modules now load seed weights from ``brain/models`` and resume their
adapters from ``brain/persistent``. Sensors stream embeddings directly through
the ``MessageBus`` and trainer updates apply sentinel-aware down-regulation at
every step. The cochlea outputs both log-mel features and token guesses in a
single pass. The hippocampus filters episodes using configurable
``recall_threshold`` and ``salience_threshold`` so only novel memories pass the
entorhinal gate【F:brain/docs/human_brain_components_reference.txt†L108-L113】. The baseline
from the subthalamic nucleus modulates norepinephrine and acetylcholine to slow
impulsive actions【F:brain/docs/human_brain_components_reference.txt†L246-L250】. Debug logs
now record this baseline together with hippocampal footprint and hormone levels
for long term analysis. SemanticFlow has been removed; speculative tokens are
kept only in the temporal lobe. Newly added persistence hooks save LoRA weights
for the amygdala, prefrontal cortex, corpus callosum, basal ganglia and
cerebellum so their neurosymbolic adaptations survive restarts. Hormone levels
now also respond to ``memory_pressure`` so serotonin rises when the hippocampus
is saturated.

## 4. Data Flow Updates

Outputs from one region remain high dimensional embeddings. For example:

```python
vision_feat = occipital_lobe(frame_emb)        # 128‑d
combined = dmn.route([vision_feat, audio_feat])# 512‑d
command = basal_ganglia.decide(combined)       # 32‑d
token_logits = motor_cortex.act(command)       # vocabulary × weights
```

Each connection mirrors the anatomical ordering described in the reference text. The thalamus filters sensor load before routing to cortical regions, the hippocampus indexes all fused vectors for later retrieval, and the cerebellum adjusts the motor plan before token emission. The corpus callosum service simply relays embeddings between hemispheric modules.

## 5. Next Steps

- Measure the impact of modality filtering and the unified Cochlea on reaction time, then refine the executive gating network accordingly【F:brain/docs/human_brain_components_reference.txt†L53-L56】.
- Stress-test the ``DistributedHippocampus`` using the new memory usage reports and refine salience gating to prevent overload【F:brain/docs/human_brain_components_reference.txt†L108-L113】.
- **Implemented:** the hippocampus now supports multiple hemispheres via the
  ``cerebral_hemispheres`` setting so each side retains its own memories while
  sharing recall through the corpus callosum【F:brain/docs/human_brain_components_reference.txt†L11-L14】.
- Evaluate the new ``memory_pressure`` hook that raises serotonin and lowers dopamine as the hippocampus fills, ensuring stable neurotransmitter levels. Hormone levels are logged alongside the subthalamic baseline for later analysis【F:brain/docs/human_brain_components_reference.txt†L246-L250】.
- Introduce a small ``serotonin_baseline`` parameter in ``HypothalamusPituitaryAxis`` so levels drift back toward typical values, preventing depressive states when novelty stays low【F:brain/docs/human_brain_components_reference.txt†L190-L208】.
- Add a matching ``dopamine_baseline`` so reward levels stabilise around 0.5 when no feedback is present.
- Extend the basal ganglia into explicit caudate/putamen/pallidus/accumbens/nigra modules so action selection aligns with biological circuits.【F:brain/docs/human_brain_components_reference.txt†L219-L245】
- Add an ``approve_action`` stage using the caudate nucleus, globus pallidus and
  subthalamic nucleus to suppress repetitive motor commands and keep the output
  varied.【F:brain/docs/human_brain_components_reference.txt†L219-L245】【F:brain/docs/human_brain_components_reference.txt†L232-L239】【F:brain/docs/human_brain_components_reference.txt†L246-L250】
- Refactor the obsolete Default Mode Network into interconnected prefrontal, posterior cingulate and angular gyrus modules.
- Introduce a curiosity-driven intrinsic motivation module so unexplored tokens receive a small bonus during selection.
- Predict candidate token valence from hippocampal memories so dopamine rises when an internally simulated action appears rewarding.
- Add a ``neurogenesis`` bootstrapping step that seeds any zero-weight region with
  Kaiming-initialized parameters and tracks which modules have been born.
- Refine the new PyGame training GUI so errors display in a dedicated pane and
  ratings are provided via clickable buttons.
- Add a small overlay showing novelty metrics alongside hormone levels in the
  viewer so fluctuations can be monitored live.
- Increase basal ganglia gating when norepinephrine or serotonin is high so the
  system vocalises when distressed and learns from early feedback.
- Delay further motor output until ratings are received or a short timeout
  expires so the system can associate feedback with the correct response.
- Profile GPU memory and inference time to better distribute regions across
  devices. The current layout (see `configs/default.yaml` lines 4–14) leaves
  `cuda:1` and `cuda:2` mostly idle while `cuda:0` and `cuda:3` are saturated.
  Rebalancing or running low-usage modules concurrently should raise throughput.
- Replace the fixed one-second sleeps in `occipital_service.py` and
  `auditory_service.py` with event-driven polling so these services react as soon
  as new sensor data arrives.

## Recent Updates

- Removed the dynamic wait in `brain.py` entirely so the main loop runs at full
  speed.
- Eliminated sleep calls in `retina.py` and `cochlea.py` to avoid throttling
  sensor updates.
- Converted `occipital_service.py` and `auditory_service.py` to wait on an event
  instead of looping with one-second sleeps, allowing immediate reaction when
  messages arrive.
- Redistributed GPU assignments in `configs/default.yaml` so sensors run on
  `cuda:1` and `cuda:2`, while the trainer now uses `cuda:0`.
- Introduced a configurable `cerebral_hemispheres` option so multiple
  hippocampal hemispheres can be enabled. Additional hemispheres operate like
  the right hemisphere, meaning Broca's area is not duplicated【F:brain/docs/human_brain_components_reference.txt†L21-L27】【F:brain/docs/human_brain_components_reference.txt†L65-L66】.
- Spoken input now updates hormone levels by comparing transcripts to the
  valence phrase table, including a new `incorrect` category that boosts
  norepinephrine when detected.
- Clamped hormone deltas and added mild decay for norepinephrine and
  acetylcholine so they no longer saturate at the maximum level.
- Replaced the ``motor_pause_seconds`` delay with a ``SupplementaryMotorArea``
  blending premotor, IFG and prefrontal signals. Its threshold ramps up over a
  configurable duration (``action_threshold_ramp_duration``, default ``60``
  seconds) so brain babbles initially but becomes quieter when dopamine
  feedback stays low. When ramping is disabled, the threshold is fixed to
  ``action_threshold_baseline`` (default ``0.75``).

This approach scales the architecture toward a more biologically faithful organisation while retaining the lightweight modular design. Each region can be trained or swapped independently, allowing experimentation with different model types without disrupting the overall system.
