"""Main integration loop linking sensors, cortical hubs and motor cortex."""

from PIL import Image
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple

from .utils.message_bus import MessageBus
from .sensors.cochlea import Cochlea
from .auditory_cortex import AuditoryCortex
from .primary_auditory_cortex import PrimaryAuditoryCortex

from .sensors.retina import Retina
from .occipital_lobe import OccipitalLobe
from .primary_visual_cortex import PrimaryVisualCortex
from .language_areas.wernickes_area import WernickesArea
from .language_areas.wernicke_adapter import WernickeAdapter
from .insular_cortex import InsularCortex
from .basal_ganglia import BasalGanglia
from .subthalamic_nucleus import SubthalamicNucleus
from .supplementary_motor_area import SupplementaryMotorArea
from .cerebellum import Cerebellum
from .corpus_callosum import CorpusCallosum
from .amygdala import Amygdala
from .frontal_lobe import FrontalLobe
from .prefrontal_cortex import PrefrontalCortex
from .inferior_parietal_lobule import InferiorParietalLobule
from .superior_parietal_lobule import SuperiorParietalLobule
from .precuneus import Precuneus
from .motor_cortex import MotorCortex
from .hypothalamus_pituitary_axis import HypothalamusPituitaryAxis
from .hippocampus import Hippocampus, DistributedHippocampus
from .subiculum import Subiculum
from .dentate_gyrus import DentateGyrus
from .thalamus import Thalamus
from .trainer import Trainer
from .temporal_lobe import TemporalLobe
from .parietal_lobe import ParietalLobe
from .entorhinal_cortex import EntorhinalCortex
from .cingulate_cortex import CingulateCortex
from .midbrain import Midbrain
from .pons import Pons
from .medulla_oblongata import MedullaOblongata
from .pituitary_gland import PituitaryGland
from .utils.config import load_config, BASE_DIR
import logging
from .utils.logger import (
    get_logger,
    enable_file_logging,
    install_handler,
    set_stdout_level,
)
from .utils import log_model_memory, log_device_memory, log_timing
from .gui_train import GUITrain
from .viewer import Viewer
from .utils.camera import Camera
from .utils.audio_buffer import AudioBuffer
from .utils.neurogenesis import maybe_initialize
from .utils.token_table import generate as generate_token_table
from .utils.valence_table import generate as generate_valence_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run brain brain")
    parser.add_argument(
        "--gui_train",
        action="store_true",
        help="enable PyGame training interface",
    )
    args = parser.parse_args(argv)

    cfg = load_config("configs/default.yaml")
    devices = cfg["devices"]
    models = cfg["models"]
    persist_dir = Path(cfg.get("persistent_dir", "persistent"))
    log_dir = Path(cfg.get("log_dir", "logs"))
    settings = cfg.get("settings", {})
    loop_interval = float(settings.get("loop_interval", 0.05))
    audio_duration = float(settings.get("audio_duration", 1.0))
    debug_no_video = bool(settings.get("debug_no_video", False))
    hippocampus_capacity = int(settings.get("hippocampus_capacity", 1000))
    recall_threshold = float(settings.get("hippocampus_recall_threshold", 0.0))
    hippocampus_shards = int(settings.get("hippocampus_shards", 1))
    cerebral_hemispheres = int(settings.get("cerebral_hemispheres", 1))
    hippocampus_independent = bool(settings.get("hippocampus_independent", False))
    num_amygdala = int(settings.get("num_amygdala", 2))
    salience_thresh = float(
        settings.get("hippocampus_salience_threshold", 0.0)
    )
    motor_candidates = int(settings.get("motor_candidates", 1))
    log_to_file = bool(settings.get("log_to_file", False))
    neurogenesis = bool(settings.get("neurogenesis", False))
    training_buffer = float(settings.get("training_buffer", 30))
    ifg_feedback_buffer = float(settings.get("ifg_feedback_buffer", 30))
    gpu_debug = bool(settings.get("gpu_debug", False))
    timing_debug = bool(settings.get("model_timing_debug", False))
    serotonin_baseline = float(settings.get("serotonin_baseline", 0.5))
    dopamine_baseline = float(settings.get("dopamine_baseline", 0.5))
    recalc_tables = bool(settings.get("recalculate_lookup_tables", False))
    enable_action_threshold_ramping = bool(
        settings.get("enable_action_threshold_ramping", False)
    )
    action_threshold_ramp_duration = float(
        settings.get("action_threshold_ramp_duration", 300.0)
    )
    action_threshold_baseline = float(
        settings.get("action_threshold_baseline", 0.75)
    )

    if not persist_dir.is_absolute():
        persist_dir = BASE_DIR / persist_dir

    if not log_dir.is_absolute():
        log_dir = BASE_DIR / log_dir

    init_state_file = persist_dir / "brain_init_state.json"

    if log_to_file:
        enable_file_logging(str(log_dir))

    logger = get_logger("brain")
    bus = MessageBus()
    models_for_profile: list[tuple[torch.nn.Module, str]] = []

    retina = Retina(models["clip"], device=devices["retina"])
    primary_vis = PrimaryVisualCortex(device=devices["occipital_lobe"])
    occipital = OccipitalLobe(device=devices["occipital_lobe"])
    cochlea = Cochlea(models["whisper"], device=devices["cochlea"])
    primary_aud = PrimaryAuditoryCortex(device=devices["auditory_cortex"])
    auditory = AuditoryCortex(device=devices["auditory_cortex"])
    if gpu_debug:
        models_for_profile.extend(
            [
                (retina.model, "retina"),
                (primary_vis, "primary_visual"),
                (occipital, "occipital_lobe"),
                (cochlea.model, "cochlea"),
                (primary_aud, "primary_auditory"),
                (auditory, "auditory_cortex"),
            ]
        )

    embedding_choice = int(cfg.get("embedding_model", 1))
    if embedding_choice == 1:
        embed_path = models["gpt2"]
    elif embedding_choice == 2:
        embed_path = models.get("bert", models["gpt2"])
    else:
        raise ValueError(f"unknown embedding_model: {embedding_choice}")

    token_table_path = persist_dir / "token_embeddings.npy"
    if recalc_tables or not token_table_path.exists():
        generate_token_table(
            embed_path,
            token_table_path,
            device=devices["language_areas"],
        )

    valence_path = persist_dir / "valence.npy"
    if recalc_tables or not valence_path.exists():
        generate_valence_table(
            embed_path,
            valence_path,
            device=devices["dmn"],
        )

    wernicke = WernickesArea(
        embed_path,
        device=devices["language_areas"],
        token_table_path=f"{persist_dir}/token_embeddings.npy",
    )
    if gpu_debug:
        models_for_profile.append((wernicke.model, "wernicke"))

    if valence_path.exists():
        table = np.load(valence_path, allow_pickle=True).item()
        like_emb = (
            torch.tensor(table.get("positive"), device=devices["dmn"])
            .mean(dim=0, keepdim=True)
        )
        dislike_emb = (
            torch.tensor(table.get("negative"), device=devices["dmn"])
            .mean(dim=0, keepdim=True)
        )
        love_emb = (
            torch.tensor(table.get("affection"), device=devices["dmn"])
            .mean(dim=0, keepdim=True)
        )
        incorrect_emb = (
            torch.tensor(table.get("incorrect"), device=devices["dmn"])
            .mean(dim=0, keepdim=True)
        )
    else:
        like_emb = torch.zeros(1, 768, device=devices["dmn"])
        dislike_emb = torch.zeros(1, 768, device=devices["dmn"])
        love_emb = torch.zeros(1, 768, device=devices["dmn"])
        incorrect_emb = torch.zeros(1, 768, device=devices["dmn"])

    sup_parietal = SuperiorParietalLobule(
        device=devices["dmn"], persist_path=f"{persist_dir}/superior_parietal_lobule.pt"
    )
    maybe_initialize(
        sup_parietal,
        f"{persist_dir}/superior_parietal_lobule.pt",
        "superior_parietal_lobule",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((sup_parietal, "superior_parietal_lobule"))
    inf_parietal = InferiorParietalLobule(
        device=devices["dmn"], persist_path=f"{persist_dir}/inferior_parietal_lobule.pt"
    )
    maybe_initialize(
        inf_parietal,
        f"{persist_dir}/inferior_parietal_lobule.pt",
        "inferior_parietal_lobule",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((inf_parietal, "inferior_parietal_lobule"))
    precuneus = Precuneus(
        input_dim=128 + 896 + 768,
        output_dim=768,
        device=devices["dmn"],
        persist_path=f"{persist_dir}/precuneus.pt",
    )
    maybe_initialize(
        precuneus,
        f"{persist_dir}/precuneus.pt",
        "precuneus",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((precuneus, "precuneus"))
    hip_dims = {
        "vision": 128,
        "audio": 896,
        "intero": 768,
        "context": 768,
        "motor": 768,
        "speech": 768,
    }
    total_shards = hippocampus_shards
    if total_shards > 1:
        shard_paths = [
            f"{persist_dir}/hippocampus_memory_shard_{i}.npz"
            for i in range(total_shards)
        ]
        hippocampus = DistributedHippocampus(
            hip_dims,
            num_shards=total_shards,
            shard_paths=shard_paths,
            independent=hippocampus_independent,
            capacity=hippocampus_capacity,
            recall_threshold=recall_threshold,
            salience_threshold=salience_thresh,
            compressed=True,
        )
    else:
        hippocampus = Hippocampus(
            dims=hip_dims,
            capacity=hippocampus_capacity,
            recall_threshold=recall_threshold,
            persist_path=f"{persist_dir}/hippocampus_memory.npz",
            salience_threshold=salience_thresh,
        )
    dentate = DentateGyrus(device=devices["dmn"], persist_path=f"{persist_dir}/dentate_gyrus.pt")
    maybe_initialize(
        dentate,
        f"{persist_dir}/dentate_gyrus.pt",
        "dentate_gyrus",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((dentate, "dentate_gyrus"))
    subiculum = Subiculum(device=devices["dmn"], persist_path=f"{persist_dir}/subiculum.pt")
    maybe_initialize(
        subiculum,
        f"{persist_dir}/subiculum.pt",
        "subiculum",
        neurogenesis,
        init_state_file,
        var_scale=1.1,
    )
    if gpu_debug:
        models_for_profile.append((subiculum, "subiculum"))
    amygdalae = []
    for idx in range(num_amygdala):
        amg = Amygdala(
            device=devices["dmn"],
            persist_path=f"{persist_dir}/amygdala_{idx}.pt",
        )
        maybe_initialize(
            amg,
            f"{persist_dir}/amygdala_{idx}.pt",
            f"amygdala_{idx}",
            neurogenesis,
            init_state_file,
        )
        if gpu_debug:
            models_for_profile.append((amg, f"amygdala_{idx}"))
        amygdalae.append(amg)

    def eval_amygdala(embedding: torch.Tensor) -> float:
        vals = [a.evaluate(embedding) for a in amygdalae]
        return float(sum(vals) / len(vals)) if vals else 0.0
    frontal = FrontalLobe(
        device=devices["dmn"],
        persist_path=f"{persist_dir}/frontal_lobe.pt",
        ifg_feedback_buffer=ifg_feedback_buffer,
    )
    maybe_initialize(
        frontal,
        f"{persist_dir}/frontal_lobe.pt",
        "frontal_lobe",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((frontal, "frontal_lobe"))
    pfc = frontal.prefrontal
    corpus = CorpusCallosum(
        embed_dim=768,
        device=devices["dmn"],
        persist_path=f"{persist_dir}/corpus_callosum_bridge.pt",
    )
    maybe_initialize(
        corpus,
        f"{persist_dir}/corpus_callosum_bridge.pt",
        "corpus_callosum_bridge",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((corpus, "corpus_callosum"))
    axis = HypothalamusPituitaryAxis(
        serotonin_baseline=serotonin_baseline,
        dopamine_baseline=dopamine_baseline,
    )
    pituitary = PituitaryGland(device=devices["dmn"], persist_path=f"{persist_dir}/pituitary_gland.pt")
    maybe_initialize(
        pituitary,
        f"{persist_dir}/pituitary_gland.pt",
        "pituitary_gland",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((pituitary, "pituitary_gland"))
    entorhinal = EntorhinalCortex(device=devices["dmn"], persist_path=f"{persist_dir}/entorhinal_cortex.pt")
    maybe_initialize(
        entorhinal,
        f"{persist_dir}/entorhinal_cortex.pt",
        "entorhinal_cortex",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((entorhinal, "entorhinal_cortex"))
    parietal = ParietalLobe(device=devices["occipital_lobe"], persist_path=f"{persist_dir}/parietal_lobe.pt")
    maybe_initialize(
        parietal,
        f"{persist_dir}/parietal_lobe.pt",
        "parietal_lobe",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((parietal, "parietal_lobe"))
    cingulate = CingulateCortex(device=devices["dmn"], persist_path=f"{persist_dir}/cingulate_cortex.pt")
    maybe_initialize(
        cingulate,
        f"{persist_dir}/cingulate_cortex.pt",
        "cingulate_cortex",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((cingulate, "cingulate_cortex"))
    midbrain = Midbrain(device=devices["dmn"], persist_path=f"{persist_dir}/midbrain.pt")
    maybe_initialize(
        midbrain,
        f"{persist_dir}/midbrain.pt",
        "midbrain",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((midbrain, "midbrain"))
    pons = Pons(device=devices["dmn"], persist_path=f"{persist_dir}/pons.pt")
    maybe_initialize(
        pons,
        f"{persist_dir}/pons.pt",
        "pons",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((pons, "pons"))
    medulla = MedullaOblongata(device=devices["dmn"], persist_path=f"{persist_dir}/medulla_oblongata.pt")
    maybe_initialize(
        medulla,
        f"{persist_dir}/medulla_oblongata.pt",
        "medulla_oblongata",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((medulla, "medulla_oblongata"))
    stn = SubthalamicNucleus(device=devices["dmn"])
    sma = SupplementaryMotorArea(
        device=devices["dmn"],
        ramp_duration=action_threshold_ramp_duration,
        target_threshold=action_threshold_baseline,
        use_ramping=enable_action_threshold_ramping,
        persist_path=f"{persist_dir}/supplementary_motor_area.pt",
    )
    maybe_initialize(
        sma,
        f"{persist_dir}/supplementary_motor_area.pt",
        "supplementary_motor_area",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((sma, "supplementary_motor_area"))
    basal = BasalGanglia(
        input_dim=768,
        device=devices["dmn"],
        axis=axis,
        prefrontal=pfc,
        premotor=frontal.premotor,
        ifg=frontal.inferior_frontal,
        supplementary=sma,
        stn=stn,
        persist_path=f"{persist_dir}/basal_ganglia_gating.pt",
        submodule_dir=str(persist_dir),
    )
    maybe_initialize(
        basal,
        f"{persist_dir}/basal_ganglia_gating.pt",
        "basal_ganglia_gating",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((basal, "basal_ganglia"))
    maybe_initialize(
        basal.caudate,
        f"{persist_dir}/caudate_nucleus.pt",
        "caudate_nucleus",
        neurogenesis,
        init_state_file,
    )
    maybe_initialize(
        basal.putamen,
        f"{persist_dir}/putamen.pt",
        "putamen",
        neurogenesis,
        init_state_file,
    )
    maybe_initialize(
        basal.pallidus,
        f"{persist_dir}/globus_pallidus.pt",
        "globus_pallidus",
        neurogenesis,
        init_state_file,
    )
    maybe_initialize(
        basal.accumbens,
        f"{persist_dir}/nucleus_accumbens.pt",
        "nucleus_accumbens",
        neurogenesis,
        init_state_file,
    )
    maybe_initialize(
        basal.nigra,
        f"{persist_dir}/substantia_nigra.pt",
        "substantia_nigra",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.extend(
            [
                (basal.caudate, "caudate_nucleus"),
                (basal.putamen, "putamen"),
                (basal.pallidus, "globus_pallidus"),
                (basal.accumbens, "nucleus_accumbens"),
                (basal.nigra, "substantia_nigra"),
            ]
        )
    insular = InsularCortex(
        device=devices["dmn"],
        persist_path=f"{persist_dir}/insular_mapping.pt",
    )
    maybe_initialize(
        insular,
        f"{persist_dir}/insular_mapping.pt",
        "insular_mapping",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((insular, "insular_cortex"))
    temporal = TemporalLobe()
    augmenter = WernickeAdapter(
        device=devices["language_areas"],
        persist_path=f"{persist_dir}/wernicke_adapter.pt",
    )
    maybe_initialize(
        augmenter,
        f"{persist_dir}/wernicke_adapter.pt",
        "wernicke_adapter",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((augmenter, "wernicke_adapter"))
    insula = InsularCortex(
        device=devices["motor_cortex"],
        persist_path=f"{persist_dir}/motor_insula.pt",
    )
    maybe_initialize(
        insula,
        f"{persist_dir}/motor_insula.pt",
        "motor_insula",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((insula, "motor_insula"))
    cerebellum = Cerebellum(
        device=devices.get("cerebellum", devices["motor_cortex"]),
        persist_path=f"{persist_dir}/cerebellum_correction.pt",
    )
    maybe_initialize(
        cerebellum,
        f"{persist_dir}/cerebellum_correction.pt",
        "cerebellum_correction",
        neurogenesis,
        init_state_file,
    )
    if gpu_debug:
        models_for_profile.append((cerebellum, "cerebellum"))
    motor = MotorCortex(
        models["gpt2"],
        wernicke,
        device=devices["motor_cortex"],
        axis=axis,
        persist_path=f"{persist_dir}/motor_cortex_adapters.pt",
        num_candidates=motor_candidates,
        feedback_buffer=training_buffer,
        basal=basal,
        ifg=frontal.inferior_frontal,
    )
    maybe_initialize(
        motor,
        f"{persist_dir}/motor_cortex_adapters.pt",
        "motor_cortex_adapters",
        False,
        init_state_file,
        bias_shift=0.01,
    )
    if gpu_debug:
        models_for_profile.append((motor.area.model.transformer, "motor_cortex"))
    motor._trainer.timing_debug = timing_debug
    motor._trainer.log_dir = log_dir

    thalamus = Thalamus()
    if gpu_debug:
        models_for_profile.append((thalamus, "thalamus"))
    trainer = Trainer()
    trainer.timing_debug = timing_debug
    trainer.log_dir = log_dir
    if gpu_debug:
        models_for_profile.append((trainer, "trainer"))
    gui = None
    if args.gui_train:
        gui = GUITrain(motor, buffer_seconds=training_buffer)
        install_handler(gui)
        set_stdout_level(logging.WARNING)

    logger.info("starting live loop; press Ctrl+C to stop")
    dmn_device = devices["dmn"]
    prev_context = None
    silent_steps = 0

    cam = None
    viewer = None
    if not debug_no_video:
        cam = Camera()
    if args.gui_train:
        viewer = gui
    elif not debug_no_video:
        viewer = Viewer(224, 224)
    audio_buf = AudioBuffer(
        samplerate=16000, channels=1, buffer_seconds=audio_duration * 2
    )

    if gpu_debug:
        for m, name in models_for_profile:
            log_model_memory(m, name, log_dir)
        for dev in set(devices.values()):
            if str(dev).startswith("cuda"):
                log_device_memory(str(dev), log_dir)

    step = 0
    pause_level = 1.0

    try:
        while True:
            step += 1
            # Adjust sensory modality weights based on hormone levels
            vis_w = 1.4 + 0.4 * axis.dopamine - 0.2 * axis.serotonin
            aud_w = 1.4 + 0.4 * axis.dopamine - 0.2 * axis.serotonin
            intero_w = (
                1.0
                + 0.5 * axis.serotonin
                - 0.2 * axis.dopamine
                - 0.1 * axis.acetylcholine
            )
            # apply sensory weighting using neurotransmitter levels

            # simple speculative step removed due to obsolete SemanticFlow
            if not debug_no_video:
                frame_bgr = cam.read()
                if frame_bgr is None:
                    logger.warning("camera frame not captured")
                    img = Image.new("RGB", (224, 224), color="white")
                    frame_rgb = np.array(img)
                else:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb).resize((224, 224))

                with log_timing("retina", "inference", timing_debug, log_dir):
                    vision_emb = retina.encode([img]).to(devices["occipital_lobe"])
                with log_timing("primary_visual_cortex", "inference", timing_debug, log_dir):
                    prim_vis = primary_vis.extract(vision_emb)
                with log_timing("occipital_lobe", "inference", timing_debug, log_dir):
                    vision_feat = occipital.process(prim_vis)
                with log_timing("parietal_lobe", "inference", timing_debug, log_dir):
                    vision_feat = parietal.attend(vision_feat)
                thalamus.submit("vision", vision_feat)
            else:
                frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                vision_feat = torch.zeros(
                    1, 128, device=devices["occipital_lobe"]
                )

            audio_np = audio_buf.read(audio_duration)
            # Compute a simple RMS volume estimate and boost the gain for display
            audio_level = float(np.sqrt(np.mean(audio_np**2))) * 10.0
            spoken = ""
            audio_feat = torch.zeros(1, 128, device=devices["auditory_cortex"])
            # Ignore near-silent audio to avoid hallucinated transcripts
            if audio_level > 0.02:
                audio_tensor = (
                    torch.from_numpy(audio_np).float().to(cochlea.device)
                )
                with log_timing("cochlea", "inference", timing_debug, log_dir):
                    spoken = cochlea.transcribe(audio_tensor)
                with log_timing("cochlea", "inference", timing_debug, log_dir):
                    emb = cochlea.encode([audio_tensor])
                with log_timing("primary_auditory_cortex", "inference", timing_debug, log_dir):
                    prim_aud = primary_aud.extract(emb)
                with log_timing("auditory_cortex", "inference", timing_debug, log_dir):
                    audio_feat = auditory.process(prim_aud)
                if audio_feat.dim() == 3:
                    audio_feat = audio_feat.mean(dim=1)
            if spoken:
                with log_timing(
                    "wernickes_area",
                    "inference",
                    timing_debug,
                    log_dir,
                ):
                    text_emb = wernicke.encode([spoken]).mean(dim=1)
                pos_v = F.cosine_similarity(text_emb, like_emb.to(text_emb.device), dim=1).item()
                neg_v = F.cosine_similarity(text_emb, dislike_emb.to(text_emb.device), dim=1).item()
                aff_v = F.cosine_similarity(text_emb, love_emb.to(text_emb.device), dim=1).item()
                incor_v = F.cosine_similarity(text_emb, incorrect_emb.to(text_emb.device), dim=1).item()
                axis.update_valence(pos_v - neg_v - incor_v, affection=aff_v)
                if incor_v > 0.0:
                    axis.penalize_incorrect(incor_v)
                temporal.clear()
            else:
                emb_dim = getattr(
                    wernicke.model.config,
                    "n_embd",
                    getattr(wernicke.model.config, "hidden_size", 768),
                )
                text_emb = torch.zeros(
                    1,
                    emb_dim,
                    device=wernicke.device,
                )
                pos_v = neg_v = aff_v = incor_v = 0.0
            user_emb = augmenter(text_emb)
            if incor_v > 0.0:
                user_emb = user_emb + incor_v * incorrect_emb.to(user_emb.device)
            spec_emb = temporal.embedding(wernicke)
            spec_emb = augmenter(spec_emb)
            if spoken:
                text_mix = 0.9 * user_emb + 0.1 * spec_emb
            else:
                text_mix = spec_emb
            combined_audio = torch.cat(
                [
                    text_mix.to(audio_feat.device),
                    audio_feat,
                ],
                dim=-1,
            )
            thalamus.submit("audio", combined_audio)

            vision = thalamus.relay("vision")
            if vision is None:
                vision = torch.zeros(1, 128, device=dmn_device)
            else:
                vision = vision.to(dmn_device)

            audio = thalamus.relay("audio")
            if audio is None:
                audio = torch.zeros(1, 896, device=dmn_device)
            else:
                audio = audio.to(dmn_device)

            intero = thalamus.relay("intero")
            if intero is None:
                intero = torch.zeros(1, 768, device=dmn_device)
            else:
                intero = intero.to(dmn_device)
                if intero.dim() == 3:
                    intero = intero.mean(dim=1)

            # Executive filtering of sensations based on previous context
            if prev_context is not None:
                weights = pfc.filter_weights(prev_context)
                vision = vision * weights["vision"]
                audio = audio * weights["audio"]
                intero = intero * weights["intero"]

            vision_proc = sup_parietal.reconcile(vision)
            vision_proc = inf_parietal.integrate(vision_proc)
            vision_proc = vision_proc * vis_w
            audio_proc = audio * aud_w
            intero_proc = intero * intero_w
            combined = torch.cat([vision_proc, audio_proc, intero_proc], dim=-1)
            context = precuneus.reflect(combined)
            context = corpus.transfer(context)

            context_np = context.squeeze(0).detach().cpu().numpy()
            bus.publish("context", context_np.tobytes())
            if gui:
                gui.add_context(context.detach())

            if prev_context is None:
                novelty = 1.0
            else:
                sim = torch.nn.functional.cosine_similarity(
                    context.view(-1), prev_context.view(-1), dim=0
                ).clamp(min=0.0)
                novelty = float(1.0 - sim.item())

            axis.step(novelty, 0.0)
            axis.norepinephrine = max(
                0.0,
                min(1.0, axis.norepinephrine + pons.boost(torch.tensor([[novelty]])))
            )
            like_sim = torch.nn.functional.cosine_similarity(
                context.view(-1), like_emb.view(-1), dim=0
            ).item()
            dislike_sim = torch.nn.functional.cosine_similarity(
                context.view(-1), dislike_emb.view(-1), dim=0
            ).item()
            love_sim = torch.nn.functional.cosine_similarity(
                context.view(-1), love_emb.view(-1), dim=0
            ).item()
            axis.update_valence(like_sim - dislike_sim, affection=love_sim)

            # Prediction step using simple similarity to like/dislike prototypes
            prev_context = context.detach()
            recalled = hippocampus.query(
                "context", context.squeeze(0).detach().cpu().numpy(), k=5
            )
            if recalled:
                if "context" in recalled:
                    recall_ctx = torch.tensor(
                        recalled["context"], device=dmn_device
                    ).unsqueeze(0)
                    recall_ctx = subiculum(recall_ctx)
                    # Prioritize new sensory context over recalled thoughts
                    context = 0.7 * context + 0.3 * recall_ctx
                if "valence" in recalled:
                    axis.update_valence(float(recalled["valence"]))
                # Push other modalities back through the thalamus for replay
                for modality in (
                    "vision",
                    "audio",
                    "intero",
                    "motor",
                    "speech",
                ):
                    if modality in recalled:
                        tensor_val = torch.tensor(
                            recalled[modality], device=dmn_device
                        ).unsqueeze(0)
                        if modality == "motor":
                            motor_intero = insular(tensor_val)
                            filtered = axis.filter_intero(motor_intero)
                            # Negate feedback to dampen repeated thoughts
                            thalamus.submit("intero", -filtered)
                        else:
                            if modality == "speech":
                                thalamus.submit("audio", tensor_val)
                            else:
                                thalamus.submit(modality, tensor_val)

            if basal.gate(context):
                def predict_fn(embs: torch.Tensor) -> torch.Tensor:
                    aug = augmenter(embs)
                    vals = []
                    for e in aug:
                        mem = hippocampus.query(
                            "speech", e.squeeze(0).detach().cpu().numpy()
                        )
                        val = float(mem.get("valence", 0.0))
                        if val == 0.0:
                            val = eval_amygdala(e.unsqueeze(0))
                        vals.append(val)
                    return torch.tensor(vals, device=embs.device)

                with log_timing("motor_cortex", "inference", timing_debug, log_dir):
                    out_text, out_emb, fb_emb, cand_embs, best_idx, cand_texts = motor.act(
                        context,
                        valence_fn=predict_fn,
                        num_candidates=2,
                    )
                cand_aug = augmenter(cand_embs)
                out_aug = cand_aug[best_idx : best_idx + 1]
                fb_aug = augmenter(fb_emb)
                out_aug = cerebellum.adjust(out_aug, vision_feat)
                if basal.approve_action(out_aug):
                    temporal.add_speculation(cand_texts)
                    temporal.consume(out_text)
                    vis_as_motor = motor.vision_to_text(
                        vision_feat.to(motor.device)
                    )
                    with log_timing("trainer", "training", timing_debug, log_dir):
                        trainer.align(
                            [cerebellum.short_lora, cerebellum.long_lora],
                            vis_as_motor.to(cerebellum.device),
                            out_aug.to(cerebellum.device),
                        )
                    with log_timing("motor_cortex", "training", timing_debug, log_dir):
                        motor.learn_from_feedback(
                            vision_feat, user_emb, cand_aug, trainer
                        )
                    basal.register_output()
                pred_vals = predict_fn(cand_embs)
                if pred_vals.numel() > best_idx and pred_vals[best_idx] > 0:
                    axis.update_valence(float(pred_vals[best_idx]))
                else:
                    out_text = ""
                    out_aug = torch.zeros(1, 768, device=devices["motor_cortex"])
                    fb_aug = torch.zeros_like(out_aug)
            else:
                out_text = ""
                out_aug = torch.zeros(1, 768, device=devices["motor_cortex"])
                fb_aug = torch.zeros_like(out_aug)
            if out_text:
                silent_steps = 0
            else:
                silent_steps += 1
                if silent_steps > 10:
                    axis.dopamine = min(1.0, axis.dopamine + 0.3)
                    axis.serotonin = max(0.0, axis.serotonin - 0.03)
                    silent_steps = 5

            insula_emb = insula(out_aug)
            valence = eval_amygdala(context)
            valence += pos_v - neg_v - incor_v
            pain_mod = cingulate.modulate(torch.tensor([[valence]]))
            axis.dopamine = max(0.0, min(1.0, axis.dopamine + midbrain.adjust(context)))
            ctx_store = entorhinal.funnel(context)
            ctx_store = dentate.encode(ctx_store)
            if incor_v > 0.0:
                ctx_store = ctx_store + incor_v * entorhinal.funnel(incorrect_emb)
            hippocampus.add_episode(
                {
                    "vision": vision.squeeze(0).detach().cpu().numpy(),
                    "audio": audio.squeeze(0).detach().cpu().numpy(),
                    "intero": intero.squeeze(0).detach().cpu().numpy(),
                    "context": ctx_store.squeeze(0).detach().cpu().numpy(),
                    "motor": insula_emb.squeeze(0).detach().cpu().numpy(),
                    "speech": user_emb.squeeze(0).detach().cpu().numpy(),
                },
                valence=valence,
                salience=novelty,
            )
            axis.update_valence(valence + pain_mod)
            stn.reinforce(valence)
            axis.adjust_inhibition(stn.baseline)
            axis.memory_pressure(hippocampus.memory_usage_gb())
            motor_intero = insular(fb_aug)
            filtered = axis.filter_intero(motor_intero)
            # Negate feedback to dampen repeated thoughts
            thalamus.submit("intero", -filtered)
            trainer.step(
                [
                    precuneus.net,
                    precuneus.short_lora,
                    precuneus.long_lora,
                    sup_parietal.net,
                    sup_parietal.short_lora,
                    sup_parietal.long_lora,
                    inf_parietal.net,
                    inf_parietal.short_lora,
                    inf_parietal.long_lora,
                    augmenter,
                    cerebellum.short_lora,
                    cerebellum.long_lora,
                    corpus.short_lora,
                    corpus.long_lora,
                    pfc.short_lora,
                    pfc.long_lora,
                    *[a.short_lora for a in amygdalae],
                    *[a.long_lora for a in amygdalae],
                    motor.damp_lora,
                    motor.long_lora,
                    insular.short_lora,
                    insular.long_lora,
                    insula.short_lora,
                    insula.long_lora,
                    dentate.short_lora,
                    dentate.long_lora,
                    subiculum.short_lora,
                    subiculum.long_lora,
                    basal.caudate.short_lora,
                    basal.caudate.long_lora,
                    basal.putamen.short_lora,
                    basal.putamen.long_lora,
                    basal.pallidus.short_lora,
                    basal.pallidus.long_lora,
                    basal.accumbens.short_lora,
                    basal.accumbens.long_lora,
                    basal.nigra.short_lora,
                    basal.nigra.long_lora,
                    pituitary.short_lora,
                    pituitary.long_lora,
                    entorhinal.short_lora,
                    entorhinal.long_lora,
                    parietal.short_lora,
                    parietal.long_lora,
                    cingulate.short_lora,
                    cingulate.long_lora,
                    midbrain.short_lora,
                    midbrain.long_lora,
                    pons.short_lora,
                    pons.long_lora,
                    medulla,
                ],
                context,
            )

            hippocampus.decay()

            if log_to_file and step % 50 == 0:
                logger.info(
                    "stn_baseline=%.3f hippo_mem=%.2fGB",
                    stn.baseline,
                    hippocampus.memory_usage_gb(),
                )
                axis.memory_pressure(hippocampus.memory_usage_gb())
                axis.log_levels(logger)

            if viewer:
                viewer.update(
                    frame_rgb,
                    out_text,
                    audio_level,
                    {
                        "dopamine": axis.dopamine,
                        "norepinephrine": axis.norepinephrine,
                        "serotonin": axis.serotonin,
                        "acetylcholine": axis.acetylcholine,
                        "oxytocin": axis.oxytocin,
                    },
                )
                taught, treat = viewer.poll_text_input()
            else:
                taught, treat = None, False
            if treat:
                axis.update_valence(1.0)
            if taught:
                with log_timing("wernickes_area", "inference", timing_debug, log_dir):
                    teach_emb = wernicke.encode([taught]).mean(dim=1)
                teach_emb = augmenter(teach_emb)
                teach_val = eval_amygdala(teach_emb)
                tokens = wernicke.tokenizer.encode(taught)
                # training data now collected directly without transition table
                ctx_store = entorhinal.funnel(teach_emb)
                ctx_store = dentate.encode(ctx_store)
                hippocampus.add_episode(
                    {
                        "motor": teach_emb.squeeze(0).detach().cpu().numpy(),
                        "speech": teach_emb.squeeze(0).detach().cpu().numpy(),
                        "context": ctx_store.squeeze(0).detach().cpu().numpy(),
                    },
                    valence=teach_val,
                    salience=1.0,
                )
                axis.update_valence(teach_val + pain_mod)
                stn.reinforce(teach_val)
                axis.adjust_inhibition(stn.baseline)
                axis.memory_pressure(hippocampus.memory_usage_gb())
                motor_intero = insular(teach_emb)
                filtered = axis.filter_intero(motor_intero)
                # Negate feedback to dampen repeated thoughts
                thalamus.submit("intero", -filtered)
                trainer.step(
                    [
                        precuneus.net,
                        precuneus.short_lora,
                        precuneus.long_lora,
                        sup_parietal.net,
                        sup_parietal.short_lora,
                        sup_parietal.long_lora,
                        inf_parietal.net,
                        inf_parietal.short_lora,
                        inf_parietal.long_lora,
                        motor.area.model.transformer,
                        augmenter,
                        cerebellum.short_lora,
                        cerebellum.long_lora,
                        corpus.short_lora,
                        corpus.long_lora,
                        pfc.short_lora,
                        pfc.long_lora,
                        *[a.short_lora for a in amygdalae],
                        *[a.long_lora for a in amygdalae],
                        motor.damp_lora,
                        motor.long_lora,
                        insular.short_lora,
                        insular.long_lora,
                        insula.short_lora,
                        insula.long_lora,
                        dentate.short_lora,
                        dentate.long_lora,
                        subiculum.short_lora,
                        subiculum.long_lora,
                        basal.caudate.short_lora,
                        basal.caudate.long_lora,
                        basal.putamen.short_lora,
                        basal.putamen.long_lora,
                        basal.pallidus.short_lora,
                        basal.pallidus.long_lora,
                        basal.accumbens.short_lora,
                        basal.accumbens.long_lora,
                        basal.nigra.short_lora,
                        basal.nigra.long_lora,
                        pituitary.short_lora,
                        pituitary.long_lora,
                        entorhinal.short_lora,
                        entorhinal.long_lora,
                        parietal.short_lora,
                        parietal.long_lora,
                        cingulate.short_lora,
                        cingulate.long_lora,
                        midbrain.short_lora,
                        midbrain.long_lora,
                        pons.short_lora,
                        pons.long_lora,
                        medulla,
                    ],
                    teach_emb,
                    lr_scale=2.0,
                )
            pause_level = 0.95 * pause_level + 0.05 * float(stn.inhibition(context))
    except KeyboardInterrupt:
        logger.info("run interrupted")
    finally:
        if cam:
            cam.release()
        if viewer:
            viewer.close()
        audio_buf.close()
        insular.save()
        insula.save()
        dentate.save()
        subiculum.save()
        hippocampus.save()
        motor.save()
        augmenter.save()
        for a in amygdalae:
            a.save()
        pfc.save()
        corpus.save()
        basal.save()
        cerebellum.save()
        precuneus.save()
        sup_parietal.save()
        inf_parietal.save()
        pituitary.save()
        entorhinal.save()
        parietal.save()
        cingulate.save()
        midbrain.save()
        pons.save()
        medulla.save()

if __name__ == "__main__":
    main()
