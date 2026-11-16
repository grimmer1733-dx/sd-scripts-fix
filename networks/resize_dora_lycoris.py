#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parche avanzado para kohya-ss/sd-scripts (networks/resize_lora.py)
Soporte completo para LyCORIS DoRA, rangos sobre-completos (r > dim),
capas Full-Diff (.diff / .diff_b), Convoluciones 3x3/1x1 asimétricas,
compresión QR-SVD exacta y asignación espectral dinámica por energía.
"""

import os
import json
import math
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redimensionador avanzado QR-SVD para modelos DoRA / LyCORIS / Kohya SDXL."
    )
    # Argumentos compatibles con networks/resize_lora.py de kohya-ss/sd-scripts
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta del archivo .safetensors de entrada (ej. /content/concat_rango_completo...safetensors)",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        required=True,
        help="Ruta del archivo .safetensors de salida redimensionado.",
    )
    parser.add_argument(
        "--new_rank",
        type=int,
        default=64,
        help="Rango máximo objetivo para capas Lineales del UNet (o techo máximo en modos dinámicos).",
    )
    parser.add_argument(
        "--new_conv_rank",
        type=int,
        default=None,
        help="Rango máximo objetivo para capas Convolucionales (Conv 3x3 y 1x1). Por defecto igual a --new_rank.",
    )
    parser.add_argument(
        "--new_te_rank",
        type=int,
        default=None,
        help="Rango máximo objetivo para Text Encoders (lora_te1 y lora_te2). Por defecto min(new_rank, rank_actual).",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32", " float"],
        help="Precisión de guardado para los pesos (alpha siempre se guarda en fp32 por compatibilidad Kohya).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo de cómputo SVD. 'auto' evita TPUs/XLA para impedir recompilación de grafos en SVD.",
    )
    parser.add_argument(
        "--dynamic_method",
        type=str,
        default="sv_fro",
        choices=["none", "sv_fro", "sv_cumulative", "sv_ratio", "knee"],
        help=(
            "Método de selección dinámica de rango por capa: "
            "'none' = rango fijo; "
            "'sv_fro' = retiene fracción de energía de Frobenius (recomendado para DoRA); "
            "'sv_cumulative' = suma acumulada de valores singulares; "
            "'sv_ratio' = cociente respecto al mayor valor singular (sigma_1 / sigma_i <= param); "
            "'knee' = umbral óptimo de ruido de Gavish-Donoho sin hiperparámetros."
        ),
    )
    parser.add_argument(
        "--dynamic_param",
        type=float,
        default=0.94,
        help=(
            "Parámetro para el método dinámico: "
            "Para 'sv_fro' o 'sv_cumulative', fracción en (0, 1] (ej. 0.94 = 94%% de energía). "
            "Para 'sv_ratio', ratio máximo respecto a sigma_1 (ej. 10.0)."
        ),
    )
    parser.add_argument(
        "--min_rank",
        type=int,
        default=8,
        help="Rango mínimo permitido por módulo cuando se usa selección dinámica.",
    )
    parser.add_argument(
        "--rank_multiple",
        type=int,
        default=8,
        help="Redondea los rangos dinámicos a múltiplos de este entero (8 o 16 optimiza TensorCores/XLA).",
    )
    parser.add_argument(
        "--preserve_alpha_scale",
        action="store_true",
        default=True,
        help="Mantiene la relación original alpha/rank por capa (1.0 en tu DoRA). Altamente recomendado.",
    )
    parser.add_argument(
        "--convert_large_diff_to_lora",
        action="store_true",
        help="Convierte matrices .diff 2D gigantes (como label_emb y time_embed) a LoRA de bajo rango si ahorra espacio.",
    )
    parser.add_argument(
        "--use_fp64_svd",
        action="store_true",
        help="Ejecuta el núcleo QR-SVD en float64 para precisión numérica extrema antes de guardar en fp16.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Número de hilos en paralelo cuando se ejecuta en CPU.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra estadísticas detalladas de compresión y error residual.",
    )
    return parser.parse_args()


def resolve_compute_device(device_arg: str) -> torch.device:
    """
    Detecta de forma segura el dispositivo de cómputo.
    En entornos Colab con TPU (PrivateUse1 renamed to 'tpu'), ejecutar linalg.svd
    en el dispositivo XLA dispara cientos de compilaciones HLO por las formas variables.
    Si no hay GPU CUDA real, forzamos CPU multihilo con LAPACK/OpenBLAS.
    """
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[AVISO] CUDA no disponible. Usando CPU multihilo segura.")
        return torch.device("cpu")
    if device_arg == "cpu":
        return torch.device("cpu")

    # Modo 'auto'
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_save_dtype(precision: str) -> torch.dtype:
    precision = precision.strip().lower()
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    return torch.float32


def round_up_multiple(val: int, multiple: int, max_val: int) -> int:
    if multiple <= 1:
        return min(val, max_val)
    rounded = int(math.ceil(val / multiple) * multiple)
    return min(rounded, max_val)


def select_optimal_rank(
    S: torch.Tensor,
    max_target_rank: int,
    min_rank: int,
    multiple: int,
    method: str,
    param: float,
    matrix_shape: tuple,
) -> tuple[int, float]:
    """
    Calcula el rango óptimo para un espectro de valores singulares S (ordenado descendentemente)
    y devuelve (nuevo_rango, fraccion_energia_frobenius_retenida).
    """
    available_rank = S.numel()
    effective_max = min(max_target_rank, available_rank)
    effective_min = min(min_rank, effective_max)

    sq_s = S * S
    total_energy = torch.sum(sq_s).item()
    if total_energy <= 1e-20:
        return effective_min, 1.0

    if method == "none":
        chosen = effective_max
    elif method == "sv_fro":
        # Energía acumulada de Frobenius: ||Delta W_r||_F^2 / ||Delta W||_F^2 >= param
        target_energy = min(max(param, 1e-4), 1.0) * total_energy
        cumsum_energy = torch.cumsum(sq_s, dim=0)
        idx = int(torch.searchsorted(cumsum_energy, target_energy).item()) + 1
        chosen = max(effective_min, min(idx, effective_max))
        chosen = round_up_multiple(chosen, multiple, effective_max)
    elif method == "sv_cumulative":
        # Suma lineal acumulada de valores singulares
        total_sv = torch.sum(S).item()
        target_sv = min(max(param, 1e-4), 1.0) * total_sv
        cumsum_sv = torch.cumsum(S, dim=0)
        idx = int(torch.searchsorted(cumsum_sv, target_sv).item()) + 1
        chosen = max(effective_min, min(idx, effective_max))
        chosen = round_up_multiple(chosen, multiple, effective_max)
    elif method == "sv_ratio":
        # Descarta valores singulares menores a sigma_1 / param
        s_max = S[0].item()
        threshold = s_max / max(param, 1.0)
        valid = int(torch.sum(S >= threshold).item())
        chosen = max(effective_min, min(valid, effective_max))
        chosen = round_up_multiple(chosen, multiple, effective_max)
    elif method == "knee":
        # Umbral asintótico óptimo de Gavish-Donoho para matrices m x n con ruido desconocido
        m, n = matrix_shape
        beta = min(m, n) / max(m, n)
        omega = 0.56 * (beta ** 3) - 0.95 * (beta ** 2) + 1.82 * beta + 1.43
        s_median = torch.median(S).item()
        tau = omega * s_median
        valid = int(torch.sum(S >= tau).item())
        chosen = max(effective_min, min(valid, effective_max))
        chosen = round_up_multiple(chosen, multiple, effective_max)
    else:
        chosen = effective_max

    retained_energy = (torch.sum(sq_s[:chosen]).item() / total_energy) if total_energy > 0 else 1.0
    return chosen, retained_energy


def two_sided_qr_svd_resize(
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    alpha_val: float,
    max_target_rank: int,
    min_rank: int,
    rank_multiple: int,
    dynamic_method: str,
    dynamic_param: float,
    preserve_alpha_scale: bool,
    compute_device: torch.device,
    compute_dtype: torch.dtype,
    save_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Factorización Ortogonal Cruzada (Two-Sided Economy QR-SVD).
    Maneja tensores Lineales 2D y Convolucionales 4D (1x1 y 3x3 asimétricos de LyCORIS),
    incluyendo rangos sobre-completos (rank > in_dim o rank > out_dim).
    """
    orig_rank = down_weight.shape[0]
    is_conv = down_weight.ndim == 4

    # Desplegar dimensiones espaciales si es convolución
    # En tu DoRA: down=(r, C_in, kh, kw) y up=(C_out, r, 1, 1)
    if is_conv:
        _, c_in, kh_d, kw_d = down_weight.shape
        c_out, _, kh_u, kw_u = up_weight.shape
        if kh_u != 1 or kw_u != 1:
            # Caso raro donde lora_up también tuviera kernel espacial > 1
            up_2d = up_weight.permute(0, 2, 3, 1).reshape(c_out * kh_u * kw_u, orig_rank)
            down_2d = down_weight.reshape(orig_rank, c_in * kh_d * kw_d)
            up_spatial_folded = True
        else:
            up_2d = up_weight.reshape(c_out, orig_rank)
            down_2d = down_weight.reshape(orig_rank, c_in * kh_d * kw_d)
            up_spatial_folded = False
    else:
        c_out, _ = up_weight.shape
        _, c_in = down_weight.shape
        up_2d = up_weight
        down_2d = down_weight
        up_spatial_folded = False

    d_out, _ = up_2d.shape
    _, d_in = down_2d.shape

    # Mover a dispositivo y precisión de cálculo
    B = up_2d.to(device=compute_device, dtype=compute_dtype)       # (d_out, r)
    A = down_2d.to(device=compute_device, dtype=compute_dtype)     # (r, d_in)

    old_scale = float(alpha_val) / float(orig_rank) if orig_rank > 0 else 1.0

    # Paso 1: QR reducida en ambos lados
    # Si orig_rank=1424 y d_out=1280, Q_B queda de (1280, 1280) y R_B de (1280, 1424)
    Q_B, R_B = torch.linalg.qr(B, mode="reduced")
    Q_A, R_A = torch.linalg.qr(A.transpose(0, 1), mode="reduced")

    # Paso 2: Núcleo pequeño M = old_scale * R_B @ R_A^T de tamaño (min(d_out, r), min(d_in, r))
    M = (R_B @ R_A.transpose(0, 1)) * old_scale

    # Paso 3: SVD exacta sobre el núcleo M
    try:
        U_m, S, Vh_m = torch.linalg.svd(M, full_matrices=False)
    except RuntimeError:
        # Fallback ultra-estable en CPU float64 con driver gesvd si una matriz está mal condicionada
        M_cpu = M.to(device="cpu", dtype=torch.float64)
        U_m, S, Vh_m = torch.linalg.svd(M_cpu, full_matrices=False)
        U_m = U_m.to(device=compute_device, dtype=compute_dtype)
        S = S.to(device=compute_device, dtype=compute_dtype)
        Vh_m = Vh_m.to(device=compute_device, dtype=compute_dtype)

    # Paso 4: Selección óptima del nuevo rango
    new_rank, retained_energy = select_optimal_rank(
        S=S,
        max_target_rank=max_target_rank,
        min_rank=min_rank,
        multiple=rank_multiple,
        method=dynamic_method,
        param=dynamic_param,
        matrix_shape=(d_out, d_in),
    )

    # Truncamiento espectral óptimo (Teorema de Eckart-Young-Mirsky)
    U_r = U_m[:, :new_rank]
    S_r = torch.clamp(S[:new_rank], min=0.0)
    Vh_r = Vh_m[:new_rank, :]

    # Definir nuevo alpha y compensar escala para que (new_alpha / new_rank) * B_new @ A_new == U_r @ diag(S_r) @ Vh_r
    if preserve_alpha_scale:
        # Conservamos la misma proporción alpha/rank del modelo original (1.0 en tu DoRA)
        target_ratio = old_scale if old_scale > 0 else 1.0
        new_alpha_float = float(new_rank) * target_ratio
    else:
        new_alpha_float = float(new_rank)
        target_ratio = 1.0

    # Distribuir simétricamente sqrt(S_r / target_ratio) entre up y down para evitar overflow en FP16
    sqrt_sv = torch.sqrt(S_r / target_ratio)

    B_new_2d = (Q_B @ U_r) * sqrt_sv.unsqueeze(0)                  # (d_out, new_rank)
    A_new_2d = sqrt_sv.unsqueeze(1) * (Vh_r @ Q_A.transpose(0, 1)) # (new_rank, d_in)

    # Replegar dimensiones espaciales si era convolución
    if is_conv:
        if up_spatial_folded:
            B_new = B_new_2d.reshape(c_out, kh_u, kw_u, new_rank).permute(0, 3, 1, 2).contiguous()
        else:
            B_new = B_new_2d.reshape(c_out, new_rank, 1, 1).contiguous()
        A_new = A_new_2d.reshape(new_rank, c_in, kh_d, kw_d).contiguous()
    else:
        B_new = B_new_2d.contiguous()
        A_new = A_new_2d.contiguous()

    # Alpha en Kohya/LyCORIS siempre debe ser escalar 0D en float32
    alpha_tensor = torch.tensor(new_alpha_float, dtype=torch.float32, device="cpu")
    B_out = B_new.to(device="cpu", dtype=save_dtype)
    A_out = A_new.to(device="cpu", dtype=save_dtype)

    stats = {
        "orig_rank": orig_rank,
        "new_rank": new_rank,
        "retained_energy": retained_energy,
        "layer_type": f"Conv {kh_d}x{kw_d}" if is_conv else "Linear",
        "orig_params": up_weight.numel() + down_weight.numel(),
        "new_params": B_out.numel() + A_out.numel(),
    }
    return B_out, A_out, alpha_tensor, stats


def compress_full_diff_2d(
    diff_tensor: torch.Tensor,
    target_rank: int,
    compute_device: torch.device,
    compute_dtype: torch.dtype,
    save_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Opcional: Convierte una matriz .diff 2D grande (ej. label_emb o time_embed)
    a un par LoRA estándar si el ahorro de parámetros supera el 35%.
    """
    if diff_tensor.ndim != 2:
        return None
    d_out, d_in = diff_tensor.shape
    eff_rank = min(target_rank, min(d_out, d_in))
    new_params = eff_rank * (d_out + d_in)
    if new_params >= 0.65 * diff_tensor.numel():
        return None

    W = diff_tensor.to(device=compute_device, dtype=compute_dtype)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    sqrt_s = torch.sqrt(torch.clamp(S[:eff_rank], min=0.0))
    B_new = (U[:, :eff_rank] * sqrt_s.unsqueeze(0)).to(device="cpu", dtype=save_dtype).contiguous()
    A_new = (sqrt_s.unsqueeze(1) * Vh[:eff_rank, :]).to(device="cpu", dtype=save_dtype).contiguous()
    alpha_t = torch.tensor(float(eff_rank), dtype=torch.float32, device="cpu")
    return B_new, A_new, alpha_t


def group_state_dict(state_dict: dict[str, torch.Tensor]):
    """
    Agrupa todas las claves del safetensors identificando módulos LoRA/DoRA completos
    y separando tensores auxiliares puros (.diff, .diff_b).
    """
    lora_modules = defaultdict(dict)
    standalone_tensors = {}

    known_suffixes = (
        ".lora_down.weight",
        ".lora_up.weight",
        ".alpha",
        ".dora_scale",
        ".lora_mid.weight",
        ".diff",
        ".diff_b",
    )

    # Primera pasada: detectar prefijos que poseen lora_down y lora_up
    lora_prefixes = set()
    for key in state_dict.keys():
        if key.endswith(".lora_down.weight"):
            prefix = key[: -len(".lora_down.weight")]
            if f"{prefix}.lora_up.weight" in state_dict:
                lora_prefixes.add(prefix)

    # Segunda pasada: clasificar cada tensor con exactitud sin perder una sola clave
    for key, tensor in state_dict.items():
        matched_lora = False
        for suffix in known_suffixes:
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                if prefix in lora_prefixes:
                    lora_modules[prefix][suffix] = tensor
                    matched_lora = True
                break
        if not matched_lora:
            standalone_tensors[key] = tensor

    return lora_modules, standalone_tensors


def main():
    args = parse_args()
    t0 = time.time()

    compute_device = resolve_compute_device(args.device)
    compute_dtype = torch.float64 if args.use_fp64_svd else torch.float32
    save_dtype = get_save_dtype(args.save_precision)

    conv_max_rank = args.new_conv_rank if args.new_conv_rank is not None else args.new_rank
    te_max_rank = args.new_te_rank if args.new_te_rank is not None else args.new_rank

    print("=" * 72)
    print(" REDIMENSIONADOR AVANZADO QR-SVD PARA LYCORIS DoRA / KOHYA SD-SCRIPTS")
    print("=" * 72)
    print(f" -> Archivo origen     : {args.model}")
    print(f" -> Archivo destino    : {args.save_to}")
    print(f" -> Dispositivo SVD    : {compute_device} (precisión interna: {compute_dtype})")
    print(f" -> Guardado pesos     : {save_dtype} (alpha en float32)")
    print(f" -> Rango máx UNet Lin : {args.new_rank} | Conv: {conv_max_rank} | TE: {te_max_rank}")
    print(f" -> Método dinámico    : {args.dynamic_method} (param={args.dynamic_param}, min={args.min_rank})")
    print("=" * 72)

    # 1. Leer metadatos originales del encabezado safetensors
    orig_metadata = {}
    with open(args.model, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header_json = json.loads(f.read(header_size).decode("utf-8"))
        if "__metadata__" in header_json and isinstance(header_json["__metadata__"], dict):
            orig_metadata = dict(header_json["__metadata__"])

    # 2. Cargar tensores en CPU (evita inicializar o saturar memoria TPU/XLA)
    print("[1/4] Cargando tensores desde safetensors en memoria CPU...")
    state_dict = load_file(args.model, device="cpu")
    orig_total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"      Tensores cargados: {len(state_dict)} | Tamaño en RAM: {orig_total_bytes / (1024**2):.2f} MB")

    # 3. Agrupar módulos DoRA/LoRA y tensores independientes (.diff / .diff_b)
    print("[2/4] Analizando topología híbrida (DoRA + Full-Diff + Bias)...")
    lora_modules, standalone_tensors = group_state_dict(state_dict)
    del state_dict  # Liberar referencias del diccionario viejo para ahorrar RAM en Colab

    print(f"      Módulos DoRA/LoRA detectados : {len(lora_modules)}")
    print(f"      Tensores auxiliares (.diff*) : {len(standalone_tensors)}")

    new_state_dict = {}
    rank_histogram = defaultdict(int)
    energy_records = []
    orig_lora_params = 0
    new_lora_params = 0

    def process_single_module(item):
        prefix, parts = item
        down_w = parts[".lora_down.weight"]
        up_w = parts[".lora_up.weight"]
        alpha_t = parts.get(".alpha", None)
        alpha_val = float(alpha_t.item()) if alpha_t is not None else float(down_w.shape[0])

        # Determinar el techo de rango específico según componente y geometría
        is_te = prefix.startswith("lora_te")
        is_conv = down_w.ndim == 4
        if is_te:
            cap_rank = te_max_rank
        elif is_conv:
            cap_rank = conv_max_rank
        else:
            cap_rank = args.new_rank

        B_out, A_out, new_alpha_t, stats = two_sided_qr_svd_resize(
            up_weight=up_w,
            down_weight=down_w,
            alpha_val=alpha_val,
            max_target_rank=cap_rank,
            min_rank=args.min_rank,
            rank_multiple=args.rank_multiple,
            dynamic_method=args.dynamic_method,
            dynamic_param=args.dynamic_param,
            preserve_alpha_scale=args.preserve_alpha_scale,
            compute_device=compute_device,
            compute_dtype=compute_dtype,
            save_dtype=save_dtype,
        )

        out_tensors = {
            f"{prefix}.lora_up.weight": B_out,
            f"{prefix}.lora_down.weight": A_out,
            f"{prefix}.alpha": new_alpha_t,
        }

        # Preservar dora_scale y sesgos/diffs acoplados al mismo módulo (ej. .diff_b en attn_to_out o ff_net)
        if ".dora_scale" in parts:
            out_tensors[f"{prefix}.dora_scale"] = parts[".dora_scale"].to(
                device="cpu", dtype=save_dtype
            ).contiguous()

        for extra_suffix in (".diff", ".diff_b", ".lora_mid.weight"):
            if extra_suffix in parts:
                out_tensors[f"{prefix}{extra_suffix}"] = parts[extra_suffix].to(
                    device="cpu", dtype=save_dtype
                ).contiguous()

        return out_tensors, stats

    # 4. Ejecutar compresión QR-SVD
    print("[3/4] Comprimiendo módulos mediante Two-Sided Economy QR-SVD...")
    items_list = list(lora_modules.items())

    if compute_device.type == "cpu" and args.workers > 1:
        # Evitar sobre-suscripción de hilos BLAS por cada worker
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // args.workers))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for out_tensors, st in tqdm(
                executor.map(process_single_module, items_list),
                total=len(items_list),
                desc="Redimensionando DoRA (CPU Paralelo)",
            ):
                new_state_dict.update(out_tensors)
                rank_histogram[st["new_rank"]] += 1
                energy_records.append(st["retained_energy"])
                orig_lora_params += st["orig_params"]
                new_lora_params += st["new_params"]
    else:
        for item in tqdm(items_list, desc=f"Redimensionando DoRA ({compute_device})"):
            out_tensors, st = process_single_module(item)
            new_state_dict.update(out_tensors)
            rank_histogram[st["new_rank"]] += 1
            energy_records.append(st["retained_energy"])
            orig_lora_params += st["orig_params"]
            new_lora_params += st["new_params"]

    lora_modules.clear()

    # 5. Procesar tensores independientes (.diff y .diff_b puros: norm, label_emb, time_embed, out, etc.)
    converted_diffs = 0
    for key, tensor in standalone_tensors.items():
        if args.convert_large_diff_to_lora and key.endswith(".diff"):
            base_prefix = key[: -len(".diff")]
            res = compress_full_diff_2d(
                diff_tensor=tensor,
                target_rank=args.new_rank,
                compute_device=compute_device,
                compute_dtype=compute_dtype,
                save_dtype=save_dtype,
            )
            if res is not None:
                B_d, A_d, alpha_d = res
                new_state_dict[f"{base_prefix}.lora_up.weight"] = B_d
                new_state_dict[f"{base_prefix}.lora_down.weight"] = A_d
                new_state_dict[f"{base_prefix}.alpha"] = alpha_d
                converted_diffs += 1
                continue

        # Mantener intacto en precisión objetivo
        if tensor.is_floating_point():
            new_state_dict[key] = tensor.to(device="cpu", dtype=save_dtype).contiguous()
        else:
            new_state_dict[key] = tensor.to(device="cpu").contiguous()

    standalone_tensors.clear()

    # 6. Actualizar metadatos Kohya / LyCORIS para que WebUI, ComfyUI y Kohya lo reconozcan sin errores
    unique_ranks = sorted(rank_histogram.keys())
    new_metadata = dict(orig_metadata)
    if len(unique_ranks) == 1:
        r_only = unique_ranks[0]
        new_metadata["ss_network_dim"] = str(r_only)
        new_metadata["ss_network_alpha"] = str(float(r_only))
    else:
        new_metadata["ss_network_dim"] = "Dynamic"
        new_metadata["ss_network_alpha"] = "Dynamic"

    # Asegurar que lycoris.kohya y dora_wd permanezcan intactos
    new_metadata["ss_network_module"] = orig_metadata.get("ss_network_module", "lycoris.kohya")
    if "ss_network_args" in orig_metadata:
        new_metadata["ss_network_args"] = orig_metadata["ss_network_args"]
    else:
        new_metadata["ss_network_args"] = json.dumps({"dora_wd": True})

    # Metadatos informativos de trazabilidad (todos strings según especificación safetensors)
    mean_energy = sum(energy_records) / max(len(energy_records), 1)
    new_metadata["modelspec.resize_method"] = f"two_sided_qr_svd_{args.dynamic_method}"
    new_metadata["modelspec.retained_frobenius_energy"] = f"{mean_energy:.6f}"

    # 7. Guardar archivo final
    print(f"[4/4] Guardando nuevo modelo en {args.save_to} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.save_to)), exist_ok=True)
    save_file(new_state_dict, args.save_to, metadata=new_metadata)

    new_total_bytes = os.path.getsize(args.save_to)
    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print(" RESUMEN DE COMPRESIÓN COMPLETADA")
    print("=" * 72)
    print(f" -> Tamaño original          : {orig_total_bytes / (1024**2):.2f} MB")
    print(f" -> Tamaño nuevo             : {new_total_bytes / (1024**2):.2f} MB "
          f"(Reducción del {(1.0 - new_total_bytes / orig_total_bytes) * 100:.2f}%)")
    print(f" -> Parámetros LoRA antes    : {orig_lora_params / 1e6:.2f} M")
    print(f" -> Parámetros LoRA ahora    : {new_lora_params / 1e6:.2f} M")
    print(f" -> Energía media retenida   : {mean_energy * 100:.3f}% (Norma de Frobenius)")
    if converted_diffs > 0:
        print(f" -> Capas .diff pasadas LoRA : {converted_diffs}")
    print(f" -> Distribución de rangos   : {dict(sorted(rank_histogram.items()))}")
    print(f" -> Tiempo total             : {elapsed:.2f} segundos")
    print("=" * 72)


if __name__ == "__main__":
    main()
