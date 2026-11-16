import argparse
import logging
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Importaciones nativas de kohya-ss/sd-scripts (opcionales pero soportadas al 100%)
try:
    from library import sai_model_spec
    import library.model_io as model_io
    from library.model_io import SS_METADATA_KEY_BASE_MODEL_VERSION, SS_METADATA_KEY_V2
    import library.model_util as model_util
    from library.utils import setup_logging
    import networks.lora as lora

    try:
        from library import sdxl_model_util
        HAS_SDXL_UTIL = True
    except ImportError:
        HAS_SDXL_UTIL = False

    setup_logging()
    HAS_KOHYA_LIB = True
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    HAS_KOHYA_LIB = False
    HAS_SDXL_UTIL = False

logger = logging.getLogger(__name__)


# ==============================================================================
# Utilidades de E/S y Normalización Universal de Claves
# Soporta: Kohya LoRA/DoRA, LyCORIS (LoCon/Full/Diff), ComfyUI Diff, PEFT/Diffusers
# ==============================================================================

KNOWN_SUFFIXES = (
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_mid.weight",
    ".dora_scale",
    ".alpha",
    ".scalar",
    ".diff_b",
    ".diff_sc",
    ".diff",
    ".lokr_w1",
    ".lokr_w2",
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
)


def str_to_dtype(precision: Optional[str]) -> Optional[torch.dtype]:
    if precision == "fp64":
        return torch.float64
    if precision in ("float", "fp32"):
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def load_safetensors_metadata(file_name: str) -> Dict[str, str]:
    if not file_name.lower().endswith(".safetensors"):
        return {}
    with safe_open(file_name, framework="pt", device="cpu") as f:
        meta = f.metadata()
        return dict(meta) if meta is not None else {}


def convert_openclip_te2_key_to_kohya(sub_path: str) -> str:
    """
    Traduce rutas internas de OpenCLIP (SDXL Text Encoder 2 en checkpoints o ComfyUI)
    al formato exacto de Kohya: lora_te2_text_model_encoder_layers_X_...
    """
    s = sub_path
    if s.startswith("transformer.resblocks."):
        s = s[len("transformer.resblocks."):]
        parts = s.split(".", 1)
        layer_idx = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        rest = (
            rest.replace("attn.out_proj", "self_attn.out_proj")
            .replace("mlp.c_fc", "mlp.fc1")
            .replace("mlp.c_proj", "mlp.fc2")
            .replace("ln_1", "layer_norm1")
            .replace("ln_2", "layer_norm2")
        )
        return f"lora_te2_text_model_encoder_layers_{layer_idx}_{rest.replace('.', '_')}"
    return f"lora_te2_{s.replace('.', '_')}"


def canonicalize_module_name(raw_mod_name: str) -> str:
    """
    Unifica cualquier prefijo de modelo (diffusion_model.*, model.diffusion_model.*,
    conditioner.embedders.*, lycoris_*, lora_unet_*, lora_te1_*, lora_te2_*)
    al espacio canónico de Kohya para que coincidan capa a capa durante el concat.
    """
    name = raw_mod_name.strip()

    # 1. Modelos Full-Diff / ComfyUI / Sgmk de UNet:
    #    "diffusion_model.input_blocks.4.1.proj_in" -> "lora_unet_input_blocks_4_1_proj_in"
    for unet_prefix in ("model.diffusion_model.", "diffusion_model.", "unet."):
        if name.startswith(unet_prefix):
            sub = name[len(unet_prefix):]
            return "lora_unet_" + sub.replace(".", "_")

    # 2. Text Encoders estilo ComfyUI / Stability SDXL / SD1.5:
    if name.startswith("conditioner.embedders.0.transformer."):
        sub = name[len("conditioner.embedders.0.transformer."):]
        return "lora_te1_" + sub.replace(".", "_")
    if name.startswith("conditioner.embedders.1.model."):
        sub = name[len("conditioner.embedders.1.model."):]
        return convert_openclip_te2_key_to_kohya(sub)
    if name.startswith("cond_stage_model.transformer."):
        sub = name[len("cond_stage_model.transformer."):]
        return "lora_te_" + sub.replace(".", "_")

    # 3. Prefijos LyCORIS nativos -> Kohya estándar para permitir cruce directo con lora_unet_*
    if name.startswith("lycoris_"):
        sub = name[len("lycoris_"):]
        if sub.startswith(("te1_", "te2_", "te_")):
            return "lora_" + sub
        if sub.startswith("text_model_"):
            return "lora_te_" + sub
        return "lora_unet_" + sub

    # 4. Si ya empieza con lora_unet_, lora_te1_, lora_te2_, lora_te_, normalizar puntos residuales
    if name.startswith(("lora_unet", "lora_te1", "lora_te2", "lora_te")):
        return name.replace(".", "_")

    return name.replace(".", "_")


def normalize_raw_key(key: str) -> str:
    """
    Normaliza sufijos de PEFT / Diffusers / ComfyUI antes de separar módulo y atributo.
    """
    new_key = key
    # Magnitud DoRA en PEFT / Diffusers
    if ".lora_magnitude_vector.default.weight" in new_key:
        new_key = new_key.replace(".lora_magnitude_vector.default.weight", ".dora_scale")
    elif ".lora_magnitude_vector.weight" in new_key:
        new_key = new_key.replace(".lora_magnitude_vector.weight", ".dora_scale")
    elif ".lora_magnitude_vector" in new_key:
        new_key = new_key.replace(".lora_magnitude_vector", ".dora_scale")

    # Pesos LoRA en PEFT / Diffusers
    if ".lora_A.default.weight" in new_key:
        new_key = new_key.replace(".lora_A.default.weight", ".lora_down.weight")
    elif ".lora_A.weight" in new_key:
        new_key = new_key.replace(".lora_A.weight", ".lora_down.weight")

    if ".lora_B.default.weight" in new_key:
        new_key = new_key.replace(".lora_B.default.weight", ".lora_up.weight")
    elif ".lora_B.weight" in new_key:
        new_key = new_key.replace(".lora_B.weight", ".lora_up.weight")

    if ".lora_linear_layer.down.weight" in new_key:
        new_key = new_key.replace(".lora_linear_layer.down.weight", ".lora_down.weight")
    if ".lora_linear_layer.up.weight" in new_key:
        new_key = new_key.replace(".lora_linear_layer.up.weight", ".lora_up.weight")

    # Diferenciales de peso/bias alternativos
    if new_key.endswith(".weight_diff"):
        new_key = new_key[: -len(".weight_diff")] + ".diff"
    elif new_key.endswith(".bias_diff"):
        new_key = new_key[: -len(".bias_diff")] + ".diff_b"

    return new_key


def load_adapter_state_dict(
    file_name: str, compute_dtype: torch.dtype
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    if os.path.splitext(file_name)[1].lower() == ".safetensors":
        sd = load_file(file_name, device="cpu")
        metadata = load_safetensors_metadata(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        metadata = {}

    normalized_sd: Dict[str, torch.Tensor] = {}
    for key, val in sd.items():
        if not isinstance(val, torch.Tensor):
            continue
        norm_key = normalize_raw_key(key)
        normalized_sd[norm_key] = val.to(dtype=compute_dtype)

    return normalized_sd, metadata


def parse_adapter_modules(
    sd: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Agrupa todas las claves por nombre canónico de módulo, soportando simultáneamente:
    - LoRA / DoRA (.lora_down.weight, .lora_up.weight, .lora_mid.weight, .alpha, .dora_scale, .scalar)
    - Full-Diff / LyCORIS Full (.diff, .diff_b, .diff_sc)
    Devuelve (modules_dict, passthrough_unknown_keys).
    """
    modules: Dict[str, Dict[str, torch.Tensor]] = {}
    unknown_keys: Dict[str, torch.Tensor] = {}

    for key, tensor in sd.items():
        matched = False
        for suffix in KNOWN_SUFFIXES:
            if key.endswith(suffix):
                raw_mod = key[: -len(suffix)]
                canon_mod = canonicalize_module_name(raw_mod)
                field = suffix[1:]  # quitar punto inicial
                modules.setdefault(canon_mod, {})[field] = tensor
                matched = True
                break
        if not matched:
            unknown_keys[key] = tensor

    return modules, unknown_keys


# ==============================================================================
# Álgebra Exacta: Convoluciones, Tucker, SVD de Tensores .diff y Magnitud DoRA
# ==============================================================================

def collapse_tucker_to_down(down: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    """
    Integra descomposición Tucker (lora_mid [r_out, r_in, kh, kw] y lora_down [r_in, in_c, 1, 1])
    en un tensor lora_down estándar [r_out, in_c, kh, kw] sin pérdida numérica.
    """
    down_2d = down.reshape(down.size(0), down.size(1))
    collapsed = torch.einsum("o r ..., r i -> o i ...", mid, down_2d)
    return collapsed.contiguous()


def compute_full_delta(
    up: torch.Tensor,
    down: torch.Tensor,
    alpha: float,
    scalar: float = 1.0,
) -> torch.Tensor:
    """
    Reconstruye el tensor denso Delta W = (alpha / r) * scalar * (B @ A)
    para capas Linear 2D y Conv 1D/2D/3D.
    """
    rank = down.size(0)
    scale = (alpha / float(rank)) * scalar

    if down.dim() == 2 and up.dim() == 2:
        return (up @ down) * scale
    elif down.dim() == 4:
        kh, kw = down.shape[2], down.shape[3]
        up_2d = up.reshape(up.size(0), up.size(1))
        if (kh, kw) == (1, 1) and up.dim() == 4 and up.shape[2:4] == (1, 1):
            down_2d = down.reshape(down.size(0), down.size(1))
            return (up_2d @ down_2d).unsqueeze(-1).unsqueeze(-1) * scale
        else:
            down_flat = down.reshape(down.size(0), -1)
            return (up_2d @ down_flat).reshape(up.size(0), down.size(1), kh, kw) * scale
    else:
        up_2d = up.reshape(up.size(0), up.size(1))
        down_flat = down.reshape(down.size(0), -1)
        return (up_2d @ down_flat).reshape(up.size(0), *down.shape[1:]) * scale


def factorize_diff_to_low_rank(
    diff_scaled: torch.Tensor,
    max_rank: Optional[int] = None,
    sv_threshold: float = 1e-8,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Descompone un tensor diferencial denso escalado (ratio * .diff) en (up_scaled, down_scaled)
    mediante SVD para poder concatenarlo directamente con bloques LoRA / DoRA.
    - Soporta Linear [out, in] y Conv2d [out, in, kh, kw].
    - Si max_rank es None o <= 0, usa el rango efectivo completo (sin pérdida).
    - Si el tensor es 1D (LayerNorm/GroupNorm) o nulo, devuelve None.
    """
    if diff_scaled.dim() not in (2, 4):
        return None

    is_conv = diff_scaled.dim() == 4
    out_dim = diff_scaled.shape[0]
    mat = diff_scaled.reshape(out_dim, -1)

    max_possible_rank = min(mat.shape[0], mat.shape[1])
    if max_possible_rank == 0:
        return None

    # SVD en precisión completa (float64/float32)
    orig_dtype = mat.dtype
    svd_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    u, s, vh = torch.linalg.svd(mat.to(dtype=svd_dtype), full_matrices=False)

    # Filtrar valores singulares nulos o por debajo del umbral numérico relativo
    if s.numel() == 0 or float(s[0].item()) == 0.0:
        return None

    cutoff = float(s[0].item()) * sv_threshold
    valid_mask = s > cutoff
    eff_rank = int(valid_mask.sum().item())
    if eff_rank == 0:
        return None

    if max_rank is not None and max_rank > 0:
        eff_rank = min(eff_rank, max_rank)

    u = u[:, :eff_rank]
    s_sqrt = torch.sqrt(s[:eff_rank])
    vh = vh[:eff_rank, :]

    up_2d = (u * s_sqrt.unsqueeze(0)).to(dtype=orig_dtype)
    down_2d = (s_sqrt.unsqueeze(1) * vh).to(dtype=orig_dtype)

    if is_conv:
        _, in_dim, kh, kw = diff_scaled.shape
        up_out = up_2d.reshape(out_dim, eff_rank, 1, 1)
        down_out = down_2d.reshape(eff_rank, in_dim, kh, kw)
        return up_out.contiguous(), down_out.contiguous()

    return up_2d.contiguous(), down_2d.contiguous()


def row_or_col_norm(weight: torch.Tensor, wd_on_out: bool = True) -> torch.Tensor:
    """
    Calcula la norma L2 exactamente igual que LyCORIS y ComfyUI (con protección eps).
    """
    norm_dims = weight.dim() - 1
    eps = torch.finfo(weight.dtype).eps
    if wd_on_out:
        return (
            weight.reshape(weight.shape[0], -1)
            .norm(p=2, dim=1, keepdim=True)
            .reshape(weight.shape[0], *([1] * norm_dims))
            + eps
        )
    else:
        return (
            weight.transpose(0, 1)
            .reshape(weight.shape[1], -1)
            .norm(p=2, dim=1, keepdim=True)
            .reshape(weight.shape[1], *([1] * norm_dims))
            .transpose(0, 1)
            + eps
        )


def apply_dora_step(
    base_w: torch.Tensor,
    delta_unscaled: torch.Tensor,
    dora_scale: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """
    Replica la inferencia DoRA exacta de LyCORIS / ComfyUI:
    W_dora = W0 + ratio * ( dora_scale * (W0 + delta) / ||W0 + delta|| - W0 )
    """
    wd_on_out = dora_scale.shape[0] == base_w.shape[0]
    v = base_w + delta_unscaled
    v_norm = row_or_col_norm(v, wd_on_out=wd_on_out)
    decomposed = v * (dora_scale.reshape_as(v_norm) / v_norm)
    if ratio == 1.0:
        return decomposed
    return base_w + ratio * (decomposed - base_w)


# ==============================================================================
# Extracción Completa de Pesos Base (SD 1.5 / SD 2.x / SDXL UNet + TE1 + TE2)
# ==============================================================================

def build_base_weight_map(
    sd_model_path: str,
    is_sdxl: bool,
    v2: bool,
    compute_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Extrae TODOS los pesos W0 del checkpoint base (incluyendo atención, MLP, Conv2d 1x1/3x3,
    time_embed, label_emb, GroupNorm, LayerNorm, TE1 y OpenCLIP TE2) mapeados a nombres Kohya.
    Funciona tanto con kohya-ss/sd-scripts como leyendo directamente un .safetensors de SDXL/SD1.5.
    """
    logger.info(f"Cargando pesos del modelo base para calibración exacta de DoRA: {sd_model_path}")
    name_to_weight: Dict[str, torch.Tensor] = {}

    # Vía 1: Si el checkpoint es un archivo .safetensors / .ckpt estándar de 1 archivo,
    # podemos mapear directamente todas sus claves (UNet completa + TE1 + TE2) sin dejarnos
    # capas fuera de TARGET_REPLACE_MODULE (como time_embed, label_emb, skip_connection u op).
    raw_sd, _ = load_adapter_state_dict(sd_model_path, torch.float32)

    mapped_direct = 0
    for k, w in raw_sd.items():
        if not k.endswith(".weight"):
            continue
        raw_mod = k[: -len(".weight")]

        # Mapear QKV fusionado de OpenCLIP TE2 en checkpoints SDXL:
        # conditioner.embedders.1.model.transformer.resblocks.X.attn.in_proj_weight (no termina en .weight)
        canon = canonicalize_module_name(raw_mod)
        if canon.startswith(("lora_unet_", "lora_te1_", "lora_te2_", "lora_te_")):
            name_to_weight[canon] = w.detach().to(device="cpu", dtype=compute_dtype)
            mapped_direct += 1

    # Soporte especial para in_proj_weight fusionado [3*dim, dim] en OpenCLIP (SDXL TE2)
    qkv_pattern = re.compile(
        r"^conditioner\.embedders\.1\.model\.transformer\.resblocks\.(\d+)\.attn\.in_proj_weight$"
    )
    for k, w in raw_sd.items():
        m = qkv_pattern.match(k)
        if m:
            layer_idx = m.group(1)
            q_w, k_w, v_w = torch.chunk(w, 3, dim=0)
            base_prefix = f"lora_te2_text_model_encoder_layers_{layer_idx}_self_attn"
            name_to_weight[f"{base_prefix}_q_proj"] = q_w.detach().to(device="cpu", dtype=compute_dtype)
            name_to_weight[f"{base_prefix}_k_proj"] = k_w.detach().to(device="cpu", dtype=compute_dtype)
            name_to_weight[f"{base_prefix}_v_proj"] = v_w.detach().to(device="cpu", dtype=compute_dtype)
            mapped_direct += 3

    del raw_sd

    # Vía 2: Respaldo con cargadores internos de Kohya si el checkpoint tenía formato Diffusers/especial
    if mapped_direct < 100 and HAS_KOHYA_LIB:
        logger.info("Usando cargador interno de Kohya para completar el mapa de pesos base...")
        if is_sdxl:
            if not HAS_SDXL_UTIL:
                raise RuntimeError("No se encontró library.sdxl_model_util en sd-scripts.")
            te1, te2, _, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
                sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, sd_model_path, "cpu"
            )
            roots = [
                (te1, "lora_te1"),
                (te2, "lora_te2"),
                (unet, "lora_unet"),
            ]
        else:
            te, _, unet = model_util.load_models_from_stable_diffusion_checkpoint(v2, sd_model_path)
            roots = [
                (te, "lora_te"),
                (unet, "lora_unet"),
            ]

        for root_module, prefix in roots:
            for mod_path, module in root_module.named_modules():
                if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                    lora_name = f"{prefix}_{mod_path.replace('.', '_')}"
                    name_to_weight.setdefault(
                        lora_name, module.weight.detach().to(device="cpu", dtype=compute_dtype)
                    )

    logger.info(f"Mapeadas {len(name_to_weight)} capas del modelo base para calibración exacta de DoRA.")
    return name_to_weight


# ==============================================================================
# Motor Principal: Concatenación Universal DoRA + DoRA + LoRA + Full-Diff
# ==============================================================================

@torch.no_grad()
def concat_adapters_universal(
    model_paths: List[str],
    ratios: List[float],
    compute_dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    base_weights: Optional[Dict[str, torch.Tensor]] = None,
    diff_svd_rank: int = 64,
    diff_mode: str = "hybrid",
    output_key_format: str = "kohya",
    shuffle: bool = False,
    align_conv_kernels: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], Optional[str], Optional[str]]:
    """
    Concatena cualquier combinación de:
      - DoRA + DoRA
      - DoRA + LoRA / LoCon (con o sin descomposición Tucker)
      - DoRA / LoRA + Full-Diff (.diff / .diff_b como los volcados de diffusion_model.*)

    Modos para tensores .diff (diff_mode):
      - 'hybrid' (recomendado): Si una capa tiene simultáneamente LoRA/DoRA y .diff, factoriza
        el .diff vía SVD y lo concatena limpio en lora_up/lora_down (recalibrando dora_scale).
        Si una capa SOLO tiene .diff (ej. GroupNorm 1D, LayerNorm, time_embed, out, etc.) y ningún
        LoRA/DoRA la pisa, conserva .diff y .diff_b exactos al 100% sin pérdida de rango.
      - 'svd_all': Factoriza también todas las capas 2D/4D exclusivas de .diff a rango `diff_svd_rank`
        (conservando en .diff únicamente vectores 1D como normas y .diff_b).
    """
    loaded_adapters = []
    v2_meta = None
    base_model_meta = None

    for path, ratio in zip(model_paths, ratios):
        logger.info(f"Cargando adaptador (ratio={ratio}): {path}")
        sd, meta = load_adapter_state_dict(path, compute_dtype)
        if meta and HAS_KOHYA_LIB:
            if v2_meta is None:
                v2_meta = meta.get(SS_METADATA_KEY_V2, None)
            if base_model_meta is None:
                base_model_meta = meta.get(SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        parsed, unknown = parse_adapter_modules(sd)
        n_lora = sum(1 for m in parsed.values() if "lora_down.weight" in m)
        n_dora = sum(1 for m in parsed.values() if "dora_scale" in m)
        n_diff = sum(1 for m in parsed.values() if "diff" in m or "diff_b" in m)

        tipo_partes = []
        if n_dora > 0:
            tipo_partes.append(f"DoRA ({n_dora} dora_scale)")
        elif n_lora > 0:
            tipo_partes.append(f"LoRA ({n_lora} capas)")
        if n_diff > 0:
            tipo_partes.append(f"Full-Diff ({n_diff} módulos .diff/.diff_b)")
        if unknown:
            logger.warning(f" -> {len(unknown)} claves no reconocidas omitidas en {os.path.basename(path)}")

        logger.info(f" -> Módulos canónicos: {len(parsed)} | Detectado: {' + '.join(tipo_partes) or 'Vacío'}")
        loaded_adapters.append((parsed, ratio))

    all_module_names = sorted({mod_name for parsed, _ in loaded_adapters for mod_name in parsed.keys()})

    merged_sd: Dict[str, torch.Tensor] = {}
    out_dims_set = set()
    dora_calibrated_count = 0
    dora_preserved_count = 0
    diff_factorized_count = 0
    diff_native_count = 0
    bias_merged_count = 0

    def format_out_mod_name(canon_name: str) -> str:
        if output_key_format == "lycoris" and canon_name.startswith("lora_unet_"):
            return "lycoris_" + canon_name[len("lora_unet_"):]
        return canon_name

    for mod_name in tqdm(all_module_names, desc="Concatenando DoRA / LoRA / Full-Diff"):
        entries = [(parsed[mod_name], ratio) for parsed, ratio in loaded_adapters if mod_name in parsed]
        out_mod = format_out_mod_name(mod_name)

        # ----------------------------------------------------------------------
        # 1. Acumular diferenciales de sesgo (.diff_b) siempre de forma exacta
        # ----------------------------------------------------------------------
        acc_diff_b: Optional[torch.Tensor] = None
        for mod_dict, ratio in entries:
            if ratio == 0.0 or "diff_b" not in mod_dict:
                continue
            b_val = mod_dict["diff_b"].to(device=device, dtype=compute_dtype) * ratio
            acc_diff_b = b_val if acc_diff_b is None else (acc_diff_b + b_val)

        if acc_diff_b is not None:
            merged_sd[f"{out_mod}.diff_b"] = acc_diff_b.cpu().contiguous()
            bias_merged_count += 1

        # ----------------------------------------------------------------------
        # 2. Analizar qué tipos de pesos contiene este módulo a través de los modelos
        # ----------------------------------------------------------------------
        has_any_lowrank = any(
            ratio != 0.0 and "lora_down.weight" in m and "lora_up.weight" in m
            for m, ratio in entries
        )

        # Acumular todos los tensores .diff presentes en este módulo
        acc_diff_w: Optional[torch.Tensor] = None
        for mod_dict, ratio in entries:
            if ratio == 0.0 or "diff" not in mod_dict:
                continue
            d_val = mod_dict["diff"].to(device=device, dtype=compute_dtype) * ratio
            if "diff_sc" in mod_dict:
                d_val = d_val * float(mod_dict["diff_sc"].item())
            acc_diff_w = d_val if acc_diff_w is None else (acc_diff_w + d_val)

        # Caso A: Solo existe .diff (sin LoRA/DoRA en esta capa) y es 1D (GroupNorm/LayerNorm)
        # o estamos en modo 'hybrid' donde preservamos .diff íntegro sin pérdida SVD
        if not has_any_lowrank and acc_diff_w is not None:
            if acc_diff_w.dim() not in (2, 4) or diff_mode == "hybrid":
                merged_sd[f"{out_mod}.diff"] = acc_diff_w.cpu().contiguous()
                diff_native_count += 1
                continue

        # ----------------------------------------------------------------------
        # 3. Preparar geometría espacial (Conv2d 1x1 vs 3x3 vs Linear 2D)
        # ----------------------------------------------------------------------
        max_kh, max_kw = 1, 1
        is_conv4d = False

        if acc_diff_w is not None and acc_diff_w.dim() == 4:
            is_conv4d = True
            max_kh = max(max_kh, acc_diff_w.shape[2])
            max_kw = max(max_kw, acc_diff_w.shape[3])

        for mod_dict, ratio in entries:
            if ratio == 0.0 or "lora_down.weight" not in mod_dict or "lora_up.weight" not in mod_dict:
                continue
            d_tmp = mod_dict["lora_down.weight"]
            if "lora_mid.weight" in mod_dict:
                d_tmp = collapse_tucker_to_down(d_tmp, mod_dict["lora_mid.weight"])
            if d_tmp.dim() == 4:
                is_conv4d = True
                max_kh = max(max_kh, d_tmp.shape[2])
                max_kw = max(max_kw, d_tmp.shape[3])

        up_parts: List[torch.Tensor] = []
        down_parts: List[torch.Tensor] = []
        dora_entries: List[Tuple[torch.Tensor, float, torch.Tensor]] = []
        additive_deltas: List[torch.Tensor] = []

        # ----------------------------------------------------------------------
        # 4. Procesar entradas LoRA / DoRA
        # ----------------------------------------------------------------------
        for mod_dict, ratio in entries:
            if ratio == 0.0:
                continue
            if "lora_down.weight" not in mod_dict or "lora_up.weight" not in mod_dict:
                continue

            up = mod_dict["lora_up.weight"].to(device=device, dtype=compute_dtype)
            down = mod_dict["lora_down.weight"].to(device=device, dtype=compute_dtype)

            if "lora_mid.weight" in mod_dict:
                mid = mod_dict["lora_mid.weight"].to(device=device, dtype=compute_dtype)
                down = collapse_tucker_to_down(down, mid)

            if is_conv4d and down.dim() == 2:
                down = down.unsqueeze(-1).unsqueeze(-1)
                up = up.unsqueeze(-1).unsqueeze(-1)

            if is_conv4d and align_conv_kernels and (down.shape[2], down.shape[3]) != (max_kh, max_kw):
                pad_h = (max_kh - down.shape[2]) // 2
                pad_w = (max_kw - down.shape[3]) // 2
                down = F.pad(down, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)

            rank = down.size(0)
            alpha_tensor = mod_dict.get("alpha", None)
            alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
            if alpha == 0.0:
                alpha = float(rank)

            scalar_tensor = mod_dict.get("scalar", None)
            scalar = float(scalar_tensor.item()) if scalar_tensor is not None else 1.0

            # Respetar exactamente escalas por capa (ej. alpha=8 / rank=16 -> 0.5 en Conv 3x3, 1.0 en Linear)
            effective_scale = ratio * (alpha / float(rank)) * scalar
            sqrt_scale = math.sqrt(abs(effective_scale))
            sign = -1.0 if effective_scale < 0 else 1.0

            up_scaled = up * sqrt_scale
            down_scaled = down * (sign * sqrt_scale)

            up_parts.append(up_scaled)
            down_parts.append(down_scaled)

            if "dora_scale" in mod_dict:
                d_scale = mod_dict["dora_scale"].to(device=device, dtype=compute_dtype)
                delta_unscaled = compute_full_delta(up, down, alpha=alpha, scalar=scalar)
                dora_entries.append((d_scale, ratio, delta_unscaled))
            else:
                if base_weights is not None and mod_name in base_weights:
                    delta_scaled = compute_full_delta(up, down, alpha=alpha, scalar=scalar) * ratio
                    additive_deltas.append(delta_scaled)

        # ----------------------------------------------------------------------
        # 5. Integrar tensor .diff en la misma capa mediante SVD hacia rango concatenado
        # ----------------------------------------------------------------------
        if acc_diff_w is not None:
            diff_for_svd = acc_diff_w
            if is_conv4d and diff_for_svd.dim() == 2:
                diff_for_svd = diff_for_svd.unsqueeze(-1).unsqueeze(-1)
            if is_conv4d and align_conv_kernels and (diff_for_svd.shape[2], diff_for_svd.shape[3]) != (max_kh, max_kw):
                pad_h = (max_kh - diff_for_svd.shape[2]) // 2
                pad_w = (max_kw - diff_for_svd.shape[3]) // 2
                diff_for_svd = F.pad(diff_for_svd, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)

            svd_pair = factorize_diff_to_low_rank(diff_for_svd, max_rank=diff_svd_rank)
            if svd_pair is not None:
                diff_up, diff_down = svd_pair
                up_parts.append(diff_up)
                down_parts.append(diff_down)
                diff_factorized_count += 1
                if base_weights is not None and mod_name in base_weights:
                    # Usamos el diferencial exacto original para calibrar dora_scale sin error de truncamiento
                    additive_deltas.append(diff_for_svd)
            else:
                # Si es 1D o degenerado, preservar como .diff
                merged_sd[f"{out_mod}.diff"] = acc_diff_w.cpu().contiguous()
                diff_native_count += 1

        if not up_parts:
            continue

        # ----------------------------------------------------------------------
        # 6. Concatenar subespacios de rango y fijar alpha_out = total_rank
        # ----------------------------------------------------------------------
        cat_up = torch.cat(up_parts, dim=1)
        cat_down = torch.cat(down_parts, dim=0)
        total_rank = cat_down.size(0)
        out_dims_set.add(total_rank)

        if shuffle:
            perm = torch.randperm(total_rank, device=cat_down.device)
            cat_down = cat_down[perm]
            cat_up = cat_up[:, perm]

        merged_sd[f"{out_mod}.lora_up.weight"] = cat_up.cpu().contiguous()
        merged_sd[f"{out_mod}.lora_down.weight"] = cat_down.cpu().contiguous()
        merged_sd[f"{out_mod}.alpha"] = torch.tensor(float(total_rank), dtype=torch.float32, device="cpu")

        # ----------------------------------------------------------------------
        # 7. Fusión / Calibración de Alta Precisión para dora_scale (1 o N DoRAs)
        # ----------------------------------------------------------------------
        if len(dora_entries) > 0:
            if base_weights is not None and mod_name in base_weights:
                w0 = base_weights[mod_name].to(device=device, dtype=compute_dtype)
                if is_conv4d and w0.dim() == 2:
                    w0 = w0.unsqueeze(-1).unsqueeze(-1)

                # Encadenar todos los DoRAs presentes (DoRA + DoRA ...) sobre W0
                w_cur = w0.clone()
                for d_scale, d_ratio, delta_unscaled in dora_entries:
                    w_cur = apply_dora_step(w_cur, delta_unscaled, d_scale, d_ratio)

                # Sumar diferenciales aditivos provenientes de LoRAs estándar y tensores .diff
                for add_delta in additive_deltas:
                    w_cur = w_cur + add_delta

                ref_dscale = dora_entries[0][0]
                wd_on_out = ref_dscale.shape[0] == w0.shape[0]
                exact_dora_scale = row_or_col_norm(w_cur, wd_on_out=wd_on_out).reshape_as(ref_dscale)

                merged_sd[f"{out_mod}.dora_scale"] = exact_dora_scale.cpu().contiguous()
                dora_calibrated_count += 1
            else:
                # Modo sin checkpoint base (Base-Free):
                if len(dora_entries) == 1:
                    d_scale, _, _ = dora_entries[0]
                    merged_sd[f"{out_mod}.dora_scale"] = d_scale.cpu().contiguous()
                else:
                    # Combinación baricéntrica exacta para múltiples DoRAs del mismo modelo base:
                    # Si hay 2 DoRAs con ratios r1, r2:
                    # - Cuando r1 + r2 = 1.0 -> media ponderada exacta r1*m1 + r2*m2.
                    # - Cuando r1 = 1.0, r2 = 1.0 -> ancla en la media m_avg ~ ||W0|| y suma desviaciones:
                    #   m_out = m_anchor + sum_i r_i * (m_i - m_anchor), evitando duplicar ||W0|| por 2x.
                    ref_shape = dora_entries[0][0].shape
                    weight_sum = sum(abs(r) for _, r, _ in dora_entries) + 1e-12
                    m_anchor = sum(abs(r) * ds.reshape(ref_shape) for ds, r, _ in dora_entries) / weight_sum
                    m_comb = m_anchor + sum(
                        r * (ds.reshape(ref_shape) - m_anchor) for ds, r, _ in dora_entries
                    )
                    # Protección física: una norma de magnitud nunca puede volverse negativa
                    m_comb = torch.clamp(m_comb, min=1e-8)
                    merged_sd[f"{out_mod}.dora_scale"] = m_comb.cpu().contiguous()
                dora_preserved_count += 1

    logger.info(
        f"Resumen de fusión -> Capas DoRA calibradas con W0: {dora_calibrated_count} | "
        f"Capas DoRA base-free: {dora_preserved_count} | "
        f"Capas .diff factorizadas al rango: {diff_factorized_count} | "
        f"Capas .diff nativas preservadas: {diff_native_count} | "
        f"Sesgos .diff_b fusionados: {bias_merged_count}"
    )

    dims_list = sorted(out_dims_set)
    dim_str = str(dims_list[0]) if len(dims_list) == 1 else "Dynamic"
    uses_lycoris_features = (
        (dora_calibrated_count + dora_preserved_count) > 0
        or diff_native_count > 0
        or bias_merged_count > 0
    )
    network_args = {"dora_wd": "True"} if (dora_calibrated_count + dora_preserved_count) > 0 else None

    if HAS_KOHYA_LIB:
        net_module = "lycoris.kohya" if uses_lycoris_features else "networks.lora"
        metadata = model_io.build_minimum_network_metadata(
            v2_meta, base_model_meta, net_module, dim_str, dim_str, network_args
        )
    else:
        metadata = {
            "ss_network_module": "lycoris.kohya" if uses_lycoris_features else "networks.lora",
            "ss_network_dim": dim_str,
            "ss_network_alpha": dim_str,
        }
        if network_args:
            metadata["ss_network_args"] = '{"dora_wd": true}'

    return merged_sd, metadata, v2_meta, base_model_meta


# ==============================================================================
# Interfaz de Línea de Comandos (CLI)
# ==============================================================================

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenador universal de alta precisión para DoRA + DoRA + LoRA + Full-Diff "
            "(soporta claves diffusion_model.*.diff/.diff_b, lora_unet_*, lora_te1_*, lora_te2_*)."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Rutas a los adaptadores LoRA, DoRA o Full-Diff (.safetensors / .pt)",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        required=True,
        help="Peso/ratio para cada modelo en el mismo orden (ej: 1.0 0.8)",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        required=True,
        help="Ruta del archivo de salida (.safetensors recomendado)",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default=None,
        help=(
            "Opcional pero MUY RECOMENDADO: ruta al checkpoint base (SDXL / SD1.5) para calibrar "
            "exactamente dora_scale al mezclar DoRA con otro DoRA, LoRA o Full-Diff."
        ),
    )
    parser.add_argument(
        "--sdxl",
        action="store_true",
        help="Indica que el modelo base es SDXL / Pony / Illustrious",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Indica que el modelo base es Stable Diffusion v2.x",
    )
    parser.add_argument(
        "--diff_mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "svd_all"],
        help=(
            "'hybrid' (por defecto): factoriza vía SVD los tensores .diff en las capas donde coexisten "
            "con LoRA/DoRA para concatenarlos limpiamente, y conserva .diff nativo sin pérdida en capas "
            "exclusivas (GroupNorm, LayerNorm, time_embed, out, etc.). "
            "'svd_all': factoriza todas las capas 2D/4D .diff a bajo rango."
        ),
    )
    parser.add_argument(
        "--diff_svd_rank",
        type=int,
        default=64,
        help=(
            "Rango máximo al factorizar tensores .diff para concatenarlos con LoRA/DoRA "
            "(usa 0 para rango exacto completo sin truncar; por defecto 64 ofrece altísima fidelidad y peso ligero)."
        ),
    )
    parser.add_argument(
        "--output_key_format",
        type=str,
        default="kohya",
        choices=["kohya", "lycoris"],
        help="Formato de prefijos de salida: 'kohya' (lora_unet_*, lora_te1_*, lora_te2_*) o 'lycoris' (lycoris_*)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp64",
        choices=["fp64", "float", "fp32", "fp16", "bf16"],
        help="Precisión interna de cómputo algebraico y SVD (por defecto fp64)",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default="fp16",
        choices=["float", "fp32", "fp16", "bf16"],
        help="Precisión de guardado del archivo final (fp16 o bf16 recomendados; .alpha se guarda en fp32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo de cálculo ('cuda' acelera enormemente la SVD de capas .diff)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Permutar aleatoriamente los canales de rango tras concatenar",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Omitir metadatos extendidos SAI ModelSpec",
    )
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    if len(args.models) != len(args.ratios):
        raise ValueError(
            "El número de archivos en --models debe coincidir exactamente con el número de valores en --ratios."
        )

    compute_dtype = str_to_dtype(args.precision) or torch.float64
    save_dtype = str_to_dtype(args.save_precision) or torch.float16
    device = torch.device(args.device)

    base_weights = None
    if args.sd_model is not None:
        base_weights = build_base_weight_map(
            sd_model_path=args.sd_model,
            is_sdxl=args.sdxl,
            v2=args.v2,
            compute_dtype=compute_dtype,
        )

    svd_rank_arg = None if args.diff_svd_rank <= 0 else args.diff_svd_rank

    state_dict, metadata, v2_meta, base_model_meta = concat_adapters_universal(
        model_paths=args.models,
        ratios=args.ratios,
        compute_dtype=compute_dtype,
        device=device,
        base_weights=base_weights,
        diff_svd_rank=svd_rank_arg,
        diff_mode=args.diff_mode,
        output_key_format=args.output_key_format,
        shuffle=args.shuffle,
    )

    # Conversión segura de tipos para guardado (.alpha siempre en fp32 para evitar overflow en rangos altos)
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        if tensor.dtype.is_floating_point:
            if key.endswith(".alpha"):
                state_dict[key] = tensor.to(torch.float32)
            else:
                state_dict[key] = tensor.to(save_dtype)

    if HAS_KOHYA_LIB:
        model_hash, legacy_hash = model_io.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        if not args.no_metadata:
            is_sdxl_flag = args.sdxl or (base_model_meta is not None and "sdxl" in base_model_meta.lower())
            is_v2_flag = args.v2 or (v2_meta == "True")
            merged_from = sai_model_spec.build_merged_from(args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_meta = sai_model_spec.build_metadata(
                state_dict,
                is_v2_flag,
                is_v2_flag,
                is_sdxl_flag,
                True,
                False,
                time.time(),
                title=title,
                merged_from=merged_from,
            )
            metadata.update(sai_meta)

    out_dir = os.path.dirname(os.path.abspath(args.save_to))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Guardando adaptador unificado en: {args.save_to}")
    if args.save_to.lower().endswith(".safetensors"):
        save_file(state_dict, args.save_to, metadata=metadata)
    else:
        torch.save(state_dict, args.save_to)
    logger.info("¡Proceso finalizado con éxito!")


if __name__ == "__main__":
    main()
