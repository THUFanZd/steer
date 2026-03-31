from __future__ import annotations

import os
import functools
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from sae_lens import SAE as SAELensSAE  # type: ignore
from sae_lens import HookedSAETransformer  # type: ignore

from transformer_lens.utils import test_prompt, tokenize_and_concatenate

try:
    from utils.dataset_process import *  # type: ignore
except ModuleNotFoundError:
    # Keep this module importable in minimal environments where dataset helpers
    # are not present; core SAE loading/inference paths do not require them.
    pass

PSEUDO_ACTIVATION = False


@dataclass
class FeatureActivationResult:
    text: str
    activation_max: float
    activation_mean: float
    activation_sum: float
    max_token_index: int
    tokens: List[str]
    per_token_activations: List[float]
    layer: int
    feature_index: int


@dataclass
class SAEConfig:

    sae_checkpoint_path: str
    model_name: str = "mistral-7b"
    target_layer: int = 15


def load_model(model_name: str, device: str, use_hooked_transformer: bool):
    try:
        if use_hooked_transformer:
            if HookedSAETransformer is not None:
                print(f"Loading HookedSAETransformer for model: {model_name}")

                model = HookedSAETransformer.from_pretrained_no_processing(
                    model_name=model_name,
                    device=str(device),
                    dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                ).to(device).eval()

                return model

            raise RuntimeError("Neither SAELens nor TransformerLens is available")

        print(f"Loading AutoModelForCausalLM for model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        model = model.to(device).eval()
        return model

    except Exception as e:
        print(f"Warning: Could not load model {model_name}: {e}")
        return None


def load_tokenizer(model_name: str):
    """
    Loads the tokenizer for the language model.
    """
    import os
    token = os.environ.get("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)#, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {model_name}: {e}")
        return None


def load_sae(sae_path: str, device: str) -> Dict[str, Any]:
    try:
        # Support SAELens URI scheme: "sae-lens://release=...;sae_id=..."
        if isinstance(sae_path, str) and sae_path.startswith("sae-lens://"):
            spec = sae_path[len("sae-lens://"):]
            parts = [p.strip() for p in spec.split(";") if p.strip()]
            kv: Dict[str, str] = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k.strip()] = v.strip()
            release = kv.get("release") or kv.get("repo") or kv.get("model")
            sae_id = kv.get("sae_id") or kv.get("path")
            if not release or not sae_id:
                print("Warning: Invalid sae-lens URI. Expected keys: release and sae_id")
                return {}
            # Load SAE and move to correct device
            print(f"Loading SAE from {release}/{sae_id} to device {device}")
            loaded = SAELensSAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=str(device),
            )  # type: ignore
            sae_obj = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
            print(f"SAE loaded on {device}")
            return {"__sae_lens_obj__": sae_obj, "__source__": "sae-lens", "release": release, "sae_id": sae_id}

        # Local checkpoint path (.npz from gemma-scope, or torch checkpoint)
        if os.path.isdir(sae_path):
            npz_candidate = os.path.join(sae_path, "params.npz")
            if os.path.exists(npz_candidate):
                sae_path = npz_candidate

        if os.path.exists(sae_path):
            if sae_path.endswith(".npz"):
                npz_data = np.load(sae_path)
                sae_data: Dict[str, Any] = {}
                for key in npz_data.files:
                    arr = npz_data[key]
                    sae_data[key] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

                # Provide alias keys used by legacy code paths.
                if "W_enc" in sae_data and "encoder.weight" not in sae_data:
                    sae_data["encoder.weight"] = sae_data["W_enc"].transpose(0, 1).contiguous()
                if "W_dec" in sae_data and "decoder.weight" not in sae_data:
                    sae_data["decoder.weight"] = sae_data["W_dec"]
                if "b_enc" in sae_data and "encoder.bias" not in sae_data:
                    sae_data["encoder.bias"] = sae_data["b_enc"]
                if "b_dec" in sae_data and "decoder.bias" not in sae_data:
                    sae_data["decoder.bias"] = sae_data["b_dec"]
                return sae_data

            sae_data = torch.load(sae_path, map_location=device)
            return sae_data
        print(f"Warning: SAE file not found at {sae_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not load SAE from {sae_path}: {e}")
        return {}


def infer_sae_layer_from_path(sae_path: str) -> Optional[int]:
    if not isinstance(sae_path, str) or not sae_path:
        return None

    candidates: List[str] = [sae_path]
    if sae_path.startswith("sae-lens://"):
        spec = sae_path[len("sae-lens://"):]
        parts = [p.strip() for p in spec.split(";") if p.strip()]
        kv: Dict[str, str] = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k.strip()] = v.strip()
        sae_id = kv.get("sae_id") or kv.get("path")
        if sae_id:
            candidates.insert(0, sae_id)

    for text in candidates:
        match = re.search(r"layer[_-]?(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


class ModelWithSAEModule:

    def __init__(
        self,
        llm_name: str,
        sae_path: str,
        sae_layer: Optional[int] = None,
        feature_index: int = -1,
        device: str = "cpu",
        context_size: int = 128,
        debug: bool = False,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        sae: Optional[Dict[str, Any]] = None,
    ):
        inferred_layer = infer_sae_layer_from_path(sae_path)
        if sae_layer is None:
            sae_layer = inferred_layer
        elif inferred_layer is not None and int(sae_layer) != inferred_layer:
            print(
                f"Warning: explicit sae_layer={sae_layer} differs from sae_path inferred layer={inferred_layer}."
            )

        self.layer = int(sae_layer) if sae_layer is not None else None
        self.debug = debug  # Store debug flag
        self.device = device
        
        self.feature_index = feature_index
            
        self.model_name = llm_name
        self.sae_path = sae_path

        # Determine if we're using SAELens (which requires HookedTransformer)
        self.use_hooked_transformer = sae_path.startswith("sae-lens://") if isinstance(sae_path, str) else False

        self.sae = sae if sae is not None else load_sae(sae_path, self.device)  # Load SAE first to check compatibility
        self.model = model if model is not None else load_model(llm_name, self.device, self.use_hooked_transformer)
        if self.model is None and self.use_hooked_transformer:
            print(
                "Warning: failed to load HookedSAETransformer model; "
                "falling back to AutoModelForCausalLM with local forward-hook intervention."
            )
            self.use_hooked_transformer = False
            self.model = load_model(llm_name, self.device, use_hooked_transformer=False)
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(llm_name)
        if self.model is None:
            raise RuntimeError(
                f"Failed to load model: {llm_name}. "
                "Check model path/name and environment dependencies."
            )
        if self.use_hooked_transformer and "__sae_lens_obj__" not in self.sae:
            print(
                "Warning: SAE-Lens object missing while hooked mode was requested; "
                "falling back to local forward-hook intervention."
            )
            self.use_hooked_transformer = False
            # Important: when disabling hooked mode, ensure model API also switches
            # from TransformerLens-style forward(...) to HF-style forward(input_ids=...).
            if isinstance(self.model, HookedTransformer):
                self.model = load_model(llm_name, self.device, use_hooked_transformer=False)
                if self.model is None:
                    raise RuntimeError(
                        "Failed to reload HuggingFace model after disabling hooked mode. "
                        "Please verify llm_name and environment dependencies."
                    )
        if self.layer is None and not self.use_hooked_transformer:
            raise ValueError(
                "sae_layer is required for non-hooked models when it cannot be inferred from sae_path."
            )
        
        self.context_size = context_size
        
        self.hook_name = None
        self.act_hook_name = None
        if isinstance(self.sae, dict) and "__sae_lens_obj__" in self.sae:
            sae_obj = self.sae["__sae_lens_obj__"]
            cfg = getattr(sae_obj, "cfg", None)
            
            # 1. 尝试直接从 cfg 获取 (支持较新版本的 sae-lens)
            if hasattr(cfg, "hook_name") and cfg.hook_name:
                self.hook_name = cfg.hook_name
            else:
                # 2. 回退到从 metadata 获取 (兼容老版本)
                metadata = getattr(cfg, "metadata", None)
                if isinstance(metadata, dict):
                    self.hook_name = metadata.get("hook_name")
                    
            # 3. 终极兜底：如果还是没拿到，根据层数手动推断 (TransformerLens 默认格式)
            if not self.hook_name and self.layer is not None:
                self.hook_name = f"blocks.{self.layer}.hook_resid_post"

            if isinstance(self.hook_name, str) and self.hook_name:
                self.act_hook_name = self.hook_name + ".hook_sae_acts_post"

    def batch_calculate_activation(self, dataset, batch_size: int = 8, return_full_info: bool = False, n_tokens: int = 10000000) -> Union[Tuple[List[float], List[str]], List[FeatureActivationResult]]:
            
        if not self.sae:
            raise RuntimeError("No SAE loaded. Cannot compute feature activations without SAE.")
        
        results = []
        
        tokens_list = dataset
        
        total_batches = (len(tokens_list) + batch_size - 1) // batch_size
        if total_batches == 0:
            print("No token chunks generated from dataset.")
            return ([], []) if not return_full_info else []
        
        print(f"Loaded {len(tokens_list)} corpus samples")
        print(
            f"Starting activation batches: {total_batches} batches "
            f"(batch_size={batch_size}, ctx={self.context_size})"
        )

        log_every = max(1, total_batches // 20)  # 鈮?every 5%
        start_time = time.time()
 
        # Process texts in batches with progress logging
        processed_batches = 0
        for i in range(0, len(tokens_list), batch_size):
            batch_texts = tokens_list[i:i + batch_size]
            batch_results = self._process_batch(batch_texts)
            results.extend(batch_results)

            processed_batches += 1
            if processed_batches % log_every == 0 or processed_batches == total_batches:
                elapsed = time.time() - start_time
                avg = elapsed / processed_batches
                remaining_batches = total_batches - processed_batches
                eta = remaining_batches * avg
                pct = processed_batches / total_batches * 100
                print(
                    f"Batch {processed_batches}/{total_batches} "
                    f"({pct:.1f}%) | elapsed {elapsed:.1f}s "
                    f"| avg {avg:.2f}s/batch | eta {eta/60:.1f}m",
                    flush=True,
                )
        
        if return_full_info:
            return results
        else:
            # Return compatibility format
            activation_list = [result.activation_max for result in results]
            text_list = [result.text for result in results]
            return activation_list, text_list
    
    def _process_batch(self, batch_tokens: List[List[int]]) -> List[FeatureActivationResult]:
        
        feature_index = self.feature_index
        
        results = []
        try:
            input_ids = torch.tensor(batch_tokens, dtype=torch.long).to(self.device)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            attention_mask = (input_ids != pad_id).long().to(self.device)

            with torch.no_grad():
                if self.use_hooked_transformer:
                    # Use TransformerLens's hook system
                    hook_name = self.hook_name
                    _, cache = self.model.run_with_cache(
                        input_ids,
                        names_filter=[hook_name]
                    )
                    layer_activations = cache[hook_name]  # [batch, seq, hidden_dim]
                else:
                    # Standard HuggingFace forward pass
                    outputs = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states

                    if self.layer >= len(hidden_states):
                        raise ValueError(f"Layer {self.layer} not available.")

                    layer_activations = hidden_states[self.layer + 1]

            # Process each sample in the batch
            if not self.sae:
                raise RuntimeError("SAE not properly loaded")

            # Encode whole batch once to avoid per-sample calls.
            if layer_activations.ndim == 2:
                layer_activations = layer_activations.unsqueeze(0)
            sae_features = self._encode_with_sae(layer_activations)  # [batch, seq, n_features]
            if sae_features is None or sae_features.ndim != 3:
                raise RuntimeError("SAE encoder output must be rank-3 [batch, seq, n_features].")
            if feature_index >= sae_features.shape[-1]:
                raise ValueError(f"Feature {feature_index} out of range")
            batch_feature_activations = sae_features[..., feature_index]  # [batch, seq]
            
            
            for batch_idx, seq_ids in enumerate(batch_tokens):

                # Get tensor for current row
                text_input_ids = input_ids[batch_idx]
                
                # Remove padding
                cur_mask = attention_mask[batch_idx].bool()
                cur_mask_cpu = cur_mask.cpu().tolist()
                
                tokens = [t for t, m in zip(tokens, cur_mask_cpu) if m]

                # Apply SAE (already computed for the batch)
                feature_activations = batch_feature_activations[batch_idx][cur_mask]
                decoded_text = self.tokenizer.decode(seq_ids, skip_special_tokens=True)

                # Create result
                result = FeatureActivationResult(
                    text=decoded_text,
                    activation_max=float(feature_activations.max().item()) if feature_activations.numel() > 0 else 0.0,
                    activation_mean=float(feature_activations.mean().item()) if feature_activations.numel() > 0 else 0.0,
                    activation_sum=float(feature_activations.sum().item()) if feature_activations.numel() > 0 else 0.0,
                    max_token_index=int(feature_activations.argmax().item()) if feature_activations.numel() > 0 else 0,
                    tokens=tokens,
                    per_token_activations=feature_activations.detach().cpu().tolist(),
                    layer=self.layer,
                    feature_index=feature_index
                )
                results.append(result)

        except Exception as e:
            print(f"Error in batch processing: {e}")
            import traceback
            traceback.print_exc()
            # Error handling: create dummy results
            for seq_ids in batch_tokens:
                # Try to decode text so the failing sample can still be identified.
                try:
                    err_text = self.tokenizer.decode(seq_ids, skip_special_tokens=True)
                except Exception:
                    err_text = "Error decoding text"
                    
                error_result = FeatureActivationResult(
                    text=err_text,
                    activation_max=0.0,
                    activation_mean=0.0,
                    activation_sum=0.0,
                    max_token_index=0,
                    tokens=[],
                    per_token_activations=[],
                    layer=self.layer,
                    feature_index=feature_index
                )
                results.append(error_result)

        return results

    def get_activation_trace(self, text: str) -> Dict[str, Any]:
        
        feature_index =  self.feature_index
        
        trace: Dict[str, Any] = {
            "tokens": [],
            "token_ids": [],
            "per_token_activation": [],
            "summary_activation": 0.0,  # max activation (primary metric)
            "summary_activation_mean": 0.0,
            "summary_activation_sum": 0.0,
            "max_token_index": 0,
            "layer_index": self.layer,
            "shapes": {},
            "raw_stats": {},
        }

        if PSEUDO_ACTIVATION or self.model is None or self.tokenizer is None:
            print("鈿狅笍Warning: PyTorch, model, or tokenizer not available.")
            # Fallback stub
            ids = [ord(c) % 256 for c in text]
            fallback_activations = [0.5] * len(ids)
            trace.update({
                "tokens": list(text),
                "token_ids": ids,
                "per_token_activation": fallback_activations,
                "summary_activation": max(fallback_activations),  # max activation
                "summary_activation_mean": sum(fallback_activations) / len(fallback_activations),
                "summary_activation_sum": sum(fallback_activations),
                "max_token_index": 0,
            })
            return trace

        # Tokenize
        # print('tokenizing...')
        # print(text)
        inputs = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=512)
        # print(type(inputs))
        # print(inputs)  # 除了input_ids，还有attention_mask
        input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore
        # print(input_ids)
        # exit()
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            if self.use_hooked_transformer:
                # Use TransformerLens's run_with_cache for HookedTransformer
                hook_name = self.hook_name
                _, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=[hook_name]
                )
                layer_activations = cache[hook_name]  # [batch, seq, hidden_dim]
            else:
                # Use standard HuggingFace forward pass
                inputs_dict = {k: v.to(self.device) for k, v in inputs.items()}
                # self.model 鏄?gemma-2-2b
                outputs = self.model(**inputs_dict, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # 所有层输出的隐藏状态。[batch, seq, d_model]
                # print(type(hidden_states))
                # print(len(hidden_states))
                # print(self.layer)
                assert isinstance(hidden_states, (list, tuple))
                h = hidden_states[self.layer]
                # print(type(h))
                # print(h.shape)  # torch.Size([1, 13, 2304])
                # print(h[0][0])

                if self.layer >= len(hidden_states):
                    return trace

                layer_activations = hidden_states[self.layer]  # [batch, seq, d_model]

            batch_size, seq_len, hidden_dim = layer_activations.shape

            # Decode tokens
            ids_cpu = input_ids[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(ids_cpu)
            trace["tokens"] = tokens
            trace["token_ids"] = ids_cpu
            trace["shapes"] = {
                "layer_activations": list(layer_activations.shape),
            }

            per_token_act: Optional[torch.Tensor] = None
            summary_activation: float = 0.0
            summary_activation_mean: float = 0.0
            summary_activation_sum: float = 0.0
            max_token_index: int = 0

            # Compute SAE feature activations if SAE present.
            if self.sae:
                try:
                    sae_features = self._encode_with_sae(layer_activations)
                    if sae_features is not None and sae_features.ndim == 3 and feature_index < sae_features.shape[-1]:
                        per_token_act = sae_features[0, :, feature_index]  # 此处运行的时候，batch为1，这里相当于squeeze操作
                        summary_activation = float(per_token_act.max().item())  # Use max for feature discovery
                        summary_activation_mean = float(per_token_act.mean().item())
                        summary_activation_sum = float(per_token_act.sum().item())
                        max_token_index = int(per_token_act.argmax().item())
                        trace["shapes"]["sae_features"] = list(sae_features.shape)
                except Exception as e:
                    if hasattr(self, 'debug') and self.debug:
                        print(f"Warning: SAE encoding failed in get_activation_trace: {e}")
                    per_token_act = None

            if per_token_act is None:
                # Fallback: hidden state norm per token
                norms = layer_activations.norm(dim=-1)[0]  # [seq]
                per_token_act = norms
                summary_activation = float(norms.max().item())  # Use max for feature discovery
                summary_activation_mean = float(norms.mean().item())
                summary_activation_sum = float(norms.sum().item())
                max_token_index = int(norms.argmax().item())

            if per_token_act is not None:
                per_token_list = [float(x) for x in per_token_act.detach().cpu().tolist()]
                trace["per_token_activation"] = per_token_list
                trace["summary_activation"] = float(round(summary_activation, 4))  # max activation (primary)
                trace["summary_activation_mean"] = float(round(summary_activation_mean, 4))
                trace["summary_activation_sum"] = float(round(summary_activation_sum, 4))
                trace["max_token_index"] = max_token_index
                
                # Comprehensive stats
                if len(per_token_list) > 0:
                    mean_val = sum(per_token_list) / len(per_token_list)
                    variance = sum((x - mean_val) ** 2 for x in per_token_list) / len(per_token_list)
                    std_val = variance ** 0.5
                    
                    trace["raw_stats"] = {
                        "min": float(min(per_token_list)),
                        "max": float(max(per_token_list)),
                        "mean": float(mean_val),
                        "sum": float(sum(per_token_list)),
                        "std": float(std_val),
                        "count": len(per_token_list),
                    }

        return trace

    def _get_transformer_blocks(self):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        for path in (
            "model.layers",
            "model.model.layers",
            "transformer.h",
            "gpt_neox.layers",
        ):
            current = self.model
            ok = True
            for part in path.split("."):
                if not hasattr(current, part):
                    ok = False
                    break
                current = getattr(current, part)
            if ok and isinstance(current, (torch.nn.ModuleList, list, tuple)):
                return current
        raise RuntimeError("Could not locate transformer blocks for forward hooking.")

    def _resolve_local_intervention_module(self):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if self.layer is None:
            raise RuntimeError("layer must be set for local intervention.")

        if self.layer <= 0:
            for path in ("model.embed_tokens", "model.model.embed_tokens", "transformer.wte"):
                current = self.model
                ok = True
                for part in path.split("."):
                    if not hasattr(current, part):
                        ok = False
                        break
                    current = getattr(current, part)
                if ok and isinstance(current, torch.nn.Module):
                    return current
            raise RuntimeError("Could not locate embedding module for layer 0 intervention.")

        blocks = self._get_transformer_blocks()
        block_index = self.layer  # - 1
        if block_index < 0 or block_index >= len(blocks):
            if 0 <= self.layer < len(blocks):
                block_index = self.layer
            else:
                raise RuntimeError(
                    f"Requested SAE layer {self.layer}, but model has {len(blocks)} blocks."
                )
        return blocks[block_index]

    def _coerce_attention_mask(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            return attention_mask.to(device=input_ids.device, dtype=torch.long)
        if self.tokenizer is None:
            return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        return (input_ids != pad_id).to(dtype=torch.long, device=input_ids.device)

    def _encode_with_sae(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.sae:
            raise RuntimeError("No SAE loaded.")

        if "__sae_lens_obj__" in self.sae:
            # print('encoding with sae-lens')
            sae_obj = self.sae["__sae_lens_obj__"]
            return sae_obj.encode(residual)
        
        # print('encoding with local sae')
        w_enc = self.sae.get("W_enc")
        if w_enc is None:
            w_enc = self.sae.get("encoder.weight")
        if w_enc is None:
            raise RuntimeError("Local SAE is missing encoder weights.")

        b_dec = self.sae.get("b_dec")
        if b_dec is None:
            b_dec = self.sae.get("decoder.bias")
        b_enc = self.sae.get("b_enc")
        if b_enc is None:
            b_enc = self.sae.get("encoder.bias")
        threshold = self.sae.get("threshold")

        residual_dtype = residual.dtype
        w_enc = w_enc.to(device=residual.device, dtype=residual_dtype)
        centered = residual
        if b_dec is not None:
            centered = centered - b_dec.to(device=residual.device, dtype=residual_dtype)

        if w_enc.ndim != 2:
            raise RuntimeError("Encoder weight must be 2D.")
        if w_enc.shape[0] == residual.shape[-1]:
            pre = torch.matmul(centered, w_enc)
        elif w_enc.shape[1] == residual.shape[-1]:
            pre = torch.matmul(centered, w_enc.transpose(0, 1))
        else:
            raise RuntimeError(
                f"Incompatible encoder shape {tuple(w_enc.shape)} for residual dim {residual.shape[-1]}."
            )

        if b_enc is not None:
            pre = pre + b_enc.to(device=pre.device, dtype=pre.dtype)
        if threshold is not None:
            pre = pre - threshold.to(device=pre.device, dtype=pre.dtype)
        return F.relu(pre)

    def _decode_with_sae(self, features: torch.Tensor) -> torch.Tensor:
        if not self.sae:
            raise RuntimeError("No SAE loaded.")

        if "__sae_lens_obj__" in self.sae:
            # print('decoding with sae-lens')
            sae_obj = self.sae["__sae_lens_obj__"]
            return sae_obj.decode(features)

        # print('decoding with local sae')
        w_dec = self.sae.get("W_dec")
        if w_dec is None:
            w_dec = self.sae.get("decoder.weight")
        if w_dec is None:
            raise RuntimeError("Local SAE is missing decoder weights.")

        b_dec = self.sae.get("b_dec")
        if b_dec is None:
            b_dec = self.sae.get("decoder.bias")

        feature_dtype = features.dtype
        w_dec = w_dec.to(device=features.device, dtype=feature_dtype)
        if w_dec.ndim != 2:
            raise RuntimeError("Decoder weight must be 2D.")
        if w_dec.shape[0] == features.shape[-1]:
            recon = torch.matmul(features, w_dec)
        elif w_dec.shape[1] == features.shape[-1]:
            recon = torch.matmul(features, w_dec.transpose(0, 1))
        else:
            raise RuntimeError(
                f"Incompatible decoder shape {tuple(w_dec.shape)} for feature dim {features.shape[-1]}."
            )

        if b_dec is not None:
            recon = recon + b_dec.to(device=recon.device, dtype=recon.dtype)
        return recon

    def _apply_feature_intervention(
        self,
        features: torch.Tensor,
        feature_index: int,
        value: Union[float, torch.Tensor],
        mode: str,
        intervention_scope: str = "all_tokens",
    ) -> torch.Tensor:
        if feature_index < 0 or feature_index >= features.shape[-1]:
            raise ValueError(f"Feature {feature_index} out of range for {features.shape[-1]} features.")
        if mode not in {"clamp", "add"}:
            raise ValueError("mode must be one of: clamp, add")
        if intervention_scope not in {"all_tokens", "last_token_only"}:
            raise ValueError("intervention_scope must be one of: all_tokens, last_token_only")

        steered = features.clone()
        target = steered[..., feature_index]
        if intervention_scope == "last_token_only":
            target = target[:, -1:]
        if torch.is_tensor(value):
            v = value.to(device=target.device, dtype=target.dtype)
            if v.ndim == 0:
                if mode == "clamp":
                    target.fill_(float(v.item()))
                else:
                    target.add_(float(v.item()))
            else:
                if v.ndim == 1 and v.shape[0] == target.shape[0]:
                    v = v[:, None]
                if v.shape != target.shape:
                    raise ValueError(
                        f"Intervention tensor shape {tuple(v.shape)} does not match target shape {tuple(target.shape)}."
                    )
                if mode == "clamp":
                    target.copy_(v)
                else:
                    target.add_(v)
        else:
            scalar = float(value)
            if mode == "clamp":
                target.fill_(scalar)
            else:
                target.add_(scalar)
        return steered

    def _build_residual_sae_intervention_hook(
        self,
        *,
        feature_index: int,
        value: Union[float, torch.Tensor],
        mode: str,
        intervention_scope: str,
    ):
        def _hook_fn(clean_act, hook):
            clean_features = self._encode_with_sae(clean_act)
            clean_recon = self._decode_with_sae(clean_features)
            steered_features = self._apply_feature_intervention(
                clean_features,
                feature_index,
                value,
                mode,
                intervention_scope=intervention_scope,
            )
            steered_recon = self._decode_with_sae(steered_features)
            return steered_recon + (clean_act - clean_recon)

        return _hook_fn

    @torch.no_grad()
    def run_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shape [batch, seq].")

        input_ids = input_ids.to(self.device)
        attention_mask = self._coerce_attention_mask(input_ids, attention_mask)

        if self.use_hooked_transformer and "__sae_lens_obj__" in self.sae:
            sae_obj = self.sae["__sae_lens_obj__"]
            return self.model.run_with_saes(input_ids, saes=[sae_obj])

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        if not hasattr(outputs, "logits") or outputs.logits is None:
            raise RuntimeError("Model output does not contain logits.")
        return outputs.logits

    @torch.no_grad()
    def run_logits_with_feature_intervention(
        self,
        input_ids: torch.Tensor,
        feature_index: int,
        value: Union[float, torch.Tensor],
        mode: str = "add",
        attention_mask: Optional[torch.Tensor] = None,
        intervention_scope: str = "all_tokens",
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shape [batch, seq].")
        if mode not in {"clamp", "add"}:
            raise ValueError("mode must be one of: clamp, add")
        if intervention_scope not in {"all_tokens", "last_token_only"}:
            raise ValueError("intervention_scope must be one of: all_tokens, last_token_only")

        input_ids = input_ids.to(self.device)
        attention_mask = self._coerce_attention_mask(input_ids, attention_mask)

        if self.use_hooked_transformer:
            if "__sae_lens_obj__" not in self.sae:
                raise RuntimeError("SAE object is required for hooked-model intervention.")
            if self.hook_name is None:
                raise RuntimeError("hook_name not available for hooked-model intervention.")

            hook_fn = self._build_residual_sae_intervention_hook(
                feature_index=feature_index,
                value=value,
                mode=mode,
                intervention_scope=intervention_scope,
            )
            return self.model.run_with_hooks(
                input_ids,
                return_type="logits",
                fwd_hooks=[(self.hook_name, hook_fn)],
            )

        target_module = self._resolve_local_intervention_module()

        def _local_hook(module, hook_inputs, hook_output):  # noqa: ARG001
            hidden = hook_output[0] if isinstance(hook_output, tuple) else hook_output
            clean_features = self._encode_with_sae(hidden)
            clean_recon = self._decode_with_sae(clean_features)
            steered_features = self._apply_feature_intervention(
                clean_features,
                feature_index,
                value,
                mode,
                intervention_scope=intervention_scope,
            )
            steered_recon = self._decode_with_sae(steered_features)
            steered_hidden = steered_recon + (hidden - clean_recon)
            if isinstance(hook_output, tuple):
                return (steered_hidden, *hook_output[1:])
            return steered_hidden

        hook_handle = target_module.register_forward_hook(_local_hook)
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            if not hasattr(outputs, "logits") or outputs.logits is None:
                raise RuntimeError("Model output does not contain logits.")
            return outputs.logits
        finally:
            hook_handle.remove()

    @torch.no_grad()
    def token_change_from_tokens(
        self,
        input_ids: torch.Tensor,
        feature_index: int,
        intervention_value: float = 10.0,
        top_k: int = 10,
        attention_mask: Optional[torch.Tensor] = None,
        skip_special_tokens: bool = True,
    ) -> Dict[str, List[Tuple[str, float, int]]]:
        if top_k <= 0:
            return {
                "amplify_top_increase": [],
                "amplify_top_decrease": [],
                "suppress_top_increase": [],
                "suppress_top_decrease": [],
            }

        clean_logits = self.run_logits(input_ids=input_ids, attention_mask=attention_mask)
        amp_logits = self.run_logits_with_feature_intervention(
            input_ids=input_ids,
            feature_index=feature_index,
            value=float(intervention_value),
            mode="add",
            attention_mask=attention_mask,
        )
        sup_logits = self.run_logits_with_feature_intervention(
            input_ids=input_ids,
            feature_index=feature_index,
            value=-float(intervention_value),
            mode="add",
            attention_mask=attention_mask,
        )

        amp_delta = (amp_logits - clean_logits).mean(dim=(0, 1)).detach().float().cpu()
        sup_delta = (sup_logits - clean_logits).mean(dim=(0, 1)).detach().float().cpu()

        amp_inc_scores = amp_delta.clone()
        amp_dec_scores = amp_delta.clone()
        sup_inc_scores = sup_delta.clone()
        sup_dec_scores = sup_delta.clone()

        if skip_special_tokens and self.tokenizer is not None and hasattr(self.tokenizer, "all_special_ids"):
            for special_id in self.tokenizer.all_special_ids:
                if 0 <= special_id < amp_delta.shape[0]:
                    amp_inc_scores[special_id] = float("-inf")
                    sup_inc_scores[special_id] = float("-inf")
                    amp_dec_scores[special_id] = float("inf")
                    sup_dec_scores[special_id] = float("inf")

        vocab_size = amp_delta.shape[0]
        top_k = min(top_k, vocab_size)

        amp_inc_vals, amp_inc_ids = torch.topk(amp_inc_scores, k=top_k)
        amp_dec_vals, amp_dec_ids = torch.topk(amp_dec_scores, k=top_k, largest=False)
        sup_inc_vals, sup_inc_ids = torch.topk(sup_inc_scores, k=top_k)
        sup_dec_vals, sup_dec_ids = torch.topk(sup_dec_scores, k=top_k, largest=False)

        def _to_triplets(ids: torch.Tensor, vals: torch.Tensor) -> List[Tuple[str, float, int]]:
            id_list = ids.tolist()
            if self.tokenizer is not None:
                toks = self.tokenizer.convert_ids_to_tokens(id_list)
            else:
                toks = [str(i) for i in id_list]
            return [(str(tok), float(val), int(tok_id)) for tok, val, tok_id in zip(toks, vals.tolist(), id_list)]

        return {
            "amplify_top_increase": _to_triplets(amp_inc_ids, amp_inc_vals),
            "amplify_top_decrease": _to_triplets(amp_dec_ids, amp_dec_vals),
            "suppress_top_increase": _to_triplets(sup_inc_ids, sup_inc_vals),
            "suppress_top_decrease": _to_triplets(sup_dec_ids, sup_dec_vals),
        }
    
    @torch.no_grad()
    def _gen_hook(self, clean_act, *args, feature: int, value=None, sae=None, **kwargs):
        """
        Manually steers the value inside an SAE activation using the basic activation
        :param clean_act: the basic activation before the SAE
        """

        if value is None:
            return sae.decode(sae.encode(clean_act))

        value_tensor = None
        if torch.is_tensor(value):
            value_tensor = value.to(device=clean_act.device, dtype=clean_act.dtype)
        elif isinstance(value, (list, tuple)):
            value_tensor = torch.tensor(value, device=clean_act.device, dtype=clean_act.dtype)

        if value_tensor is not None and value_tensor.ndim > 0:
            if value_tensor.ndim == 1:
                value_tensor = value_tensor[:, None]
            if sae is None:  # transluce
                clean_act[:, :, feature] = value_tensor
                return clean_act
            encoded_act = sae.encode(clean_act)
            dirty_act = sae.decode(encoded_act)
            error_term = clean_act - dirty_act
            encoded_act[:, :, feature] = value_tensor
            hooked_act = sae.decode(encoded_act) + error_term
            return hooked_act

        if value_tensor is not None:
            value = float(value_tensor.item())

        if sae is None:  # transluce
            clean_act[:, :, feature] = value
            return clean_act

        encoded_act = sae.encode(clean_act)
        dirty_act = sae.decode(encoded_act)
        error_term = clean_act - dirty_act

        encoded_act[:, :, feature] = value
        hooked_act = sae.decode(encoded_act) + error_term

        return hooked_act

    @staticmethod
    def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)

    @staticmethod
    def set_feature_act_kl_hook(act, hook, feature: int, value):
        act[:,:,feature] = value

    @torch.no_grad()
    def _compute_kl_for_value(
        self,
        tokens: torch.Tensor,
        value: float,
        feature_index: int,
        sae_obj: SAELensSAE,
    ) -> float:
        _ = sae_obj
        attention_mask = (tokens != 0).long()
        clean_logits = self.run_logits(tokens, attention_mask=attention_mask)
        hooked_logits = self.run_logits_with_feature_intervention(
            input_ids=tokens,
            feature_index=feature_index,
            value=float(value),
            mode="clamp",
            attention_mask=attention_mask,
            intervention_scope="all_tokens",
        )
        hooked_probs = hooked_logits.softmax(dim=-1)
        clean_probs = clean_logits.softmax(dim=-1)

        clean_probs[tokens == 0] = 0
        hooked_probs[tokens == 0] = 0

        kl = self._kl_divergence(clean_probs, hooked_probs)
        means: List[float] = []
        for row in kl:
            means.append(row[row != 0].mean().item())
        return float(np.mean(means))

    @torch.no_grad()
    def _find_clamp_values_for_kl(
        self,
        tokens: torch.Tensor,
        feature_index: int,
        sae_obj: SAELensSAE,
        target_kl: float,
        max_value: float = 1000.0,
        tolerance: float = 0.1,
        max_steps: int = 12,
    ) -> Tuple[List[float], List[float]]:
        if tokens.numel() == 0:
            return [], []
        _ = max_value, max_steps
        neg = float(target_kl) < 0
        target_kl = abs(float(target_kl))

        low, high = (-1000, -1) if neg else (1, 1000)
        kl_val = -1
        mid = 0
        while (low + 1 < high) and (kl_val < target_kl or kl_val > target_kl + tolerance):
            mid = (low + high) // 2
            kl_val = self._compute_kl_for_value(tokens, mid, feature_index, sae_obj)

            if (neg and kl_val < target_kl) or (not neg and kl_val > target_kl):
                high = mid
            else:
                low = mid

        clamp_values = [float(mid)] * tokens.shape[0]
        kl_values = [float(kl_val)] * tokens.shape[0]
        return clamp_values, kl_values

    @torch.no_grad()
    def _generate_baseline_from_tokens(
        self,
        prompts_tokens: torch.Tensor,
        *,
        max_new_tokens: int,
        verbose: bool = False,
        temperature: float = 0.75,
    ) -> Tuple[List[str], List[str]]:
        
        sae_obj = self.sae["__sae_lens_obj__"]
        
        self.model.reset_hooks()
        self.model.add_hook(self.hook_name, functools.partial(self._gen_hook, feature=None, value=None, sae=sae_obj))
        outputs = self.model.generate(
            prompts_tokens,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
            temperature=temperature,
        )
        self.model.reset_hooks()

        outputs_text = self.model.to_string(outputs)
        baseline_completion = [
            x[len(self.model.to_string(prompts_tokens[i])) :] for i, x in enumerate(outputs_text)
        ]
        baseline_completion = [x.replace("\n", "\\n").replace("\r", "\\r") for x in baseline_completion]
        baseline_full = [x.replace("\n", "\\n").replace("\r", "\\r") for x in outputs_text]
        return baseline_completion, baseline_full

    @torch.no_grad()
    def generate_baseline_completions(
        self,
        prompts: List[str],
        max_new_tokens: int = 25,
        verbose: bool = False,
        temperature: float = 0.75,
    ) -> Dict[str, Any]:
        """Generate baseline completions without steering."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded for generation.")

        max_new_tokens = min(int(max_new_tokens), 25)
        if max_new_tokens <= 0:
            max_new_tokens = 1

        prompts_tokens = self.model.to_tokens(prompts)
        baseline_completion, baseline_full = self._generate_baseline_from_tokens(
            prompts_tokens,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
            temperature=temperature,
        )
        return {
            "prompts": prompts,
            "baseline_completion": baseline_completion,
            "baseline_full": baseline_full,
        }

    @torch.no_grad()
    def generate_steered_completions(
        self,
        prompts: List[str],
        feature_index : Optional[int] = None, # Attention: Allow override of feature index
        max_new_tokens: int = 25,
        verbose=False, 
        temperature=0.75,
        target_kl: Optional[float] = None,
        kl_tolerance: float = 0.1,
        kl_max_steps: int = 12,
    ) -> Dict[str, Any]:
        """Generate baseline vs steered completions with KL-guided clamp search (signed target_kl sets direction)."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded for steering.")
        if not self.sae or "__sae_lens_obj__" not in self.sae:
            raise RuntimeError("SAE must be loaded for steering.")
        if not self.use_hooked_transformer:
            raise RuntimeError("Steering requires HookedSAETransformer (sae-lens).")
        
        if feature_index is None:
            print("Attention: feature_index not provided, using default from initialization.")
            feature_index = int(self.feature_index)

        max_new_tokens = min(int(max_new_tokens), 25)
        if max_new_tokens <= 0:
            max_new_tokens = 1

        sae_obj = self.sae["__sae_lens_obj__"]

        prompts_tokens = self.model.to_tokens(prompts)

        if target_kl is None:
            raise ValueError("target_kl is required and must be signed (positive or negative).")
        target_kl = float(target_kl)
        if target_kl == 0:
            raise ValueError("target_kl must be non-zero and signed (positive or negative).")
        clamp_values, kl_values = self._find_clamp_values_for_kl(
            prompts_tokens,
            feature_index,
            sae_obj,
            target_kl=target_kl,
            tolerance=kl_tolerance,
            max_steps=kl_max_steps,
        )

        clamp_tensor = torch.tensor(clamp_values, device=prompts_tokens.device, dtype=torch.float32)
        self.model.reset_hooks()
        self.model.add_hook(self.hook_name, functools.partial(self._gen_hook, feature=feature_index, value=clamp_tensor, sae=sae_obj))
        steered_outputs = self.model.generate(prompts_tokens, max_new_tokens=max_new_tokens, verbose=verbose, temperature=temperature)
        self.model.reset_hooks()
        
        steered_completion = [x[len(self.model.to_string(prompts_tokens[i])):] for i, x in enumerate(self.model.to_string(steered_outputs))]
        steered_completion = [x.replace("\n", "\\n").replace("\r", "\\r") for x in steered_completion]
        steered_full = [x.replace("\n", "\\n").replace("\r", "\\r") for x in self.model.to_string(steered_outputs)]
        

        result = {
            "prompts": prompts,
            "steered_completion": steered_completion,
            "steered_full": steered_full,
        }
        result["clamp_values"] = clamp_values
        if kl_values is not None:
            result["kl_values"] = kl_values
            result["target_kl"] = float(target_kl)

        return result


__all__ = [
    "SAEConfig",
    "ModelWithSAEModule",
    "load_model",
    "load_tokenizer",
    "load_sae",
]


if __name__ == "__main__":
    llm_name = "google/gemma-2-2b"
    
    # 确保 sae_path 中的 layer_6 和下方的 sae_layer=6 保持一致
    sae_path = "sae-lens://release=gemma-scope-2b-pt-res;sae_id=layer_6/width_16k/average_l0_70"
    test_feature_index = 0 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"⏳ 正在初始化 ModelWithSAEModule (设备: {device})...")
    module = ModelWithSAEModule(
        llm_name=llm_name,
        sae_path=sae_path,
        sae_layer=6,
        feature_index=test_feature_index,
        device=device,
        debug=True
    )

    prompt = "Hello, world! This is a simple prompt to test SAE forward pass."
    print(f"\n🚀 开始执行前向传播并提取特征激活...\nPrompt: '{prompt}'")

    # 【修改点2】去掉了 try...except 捕获，让系统原生的报错 Traceback 直接暴露出来
    trace_result = module.get_activation_trace(prompt)
    
    print("\n" + "="*40)
    print("🎯 激活追踪结果 (Activation Trace)")
    print("="*40)
    
    tokens = trace_result.get("tokens", [])
    activations = trace_result.get("per_token_activation", [])
    
    print(f"监测特征 ID: {test_feature_index}")
    print(f"最大激活值 (Max): {trace_result.get('summary_activation')}")
    print(f"平均激活值 (Mean): {trace_result.get('summary_activation_mean')}")
    print(f"最大激活对应的 Token 索引: {trace_result.get('max_token_index')}")
    print("\n📊 逐 Token 激活详情:")
    
    if tokens and activations and len(tokens) == len(activations):
        for i, (tok, act) in enumerate(zip(tokens, activations)):
            marker = " <--- MAX" if i == trace_result.get("max_token_index") else ""
            print(f"  [{i:02d}] {tok:>15} : {act:.4f}{marker}")
    else:
        print("⚠️ 未能正确获取 tokens 或激活值列表。")
