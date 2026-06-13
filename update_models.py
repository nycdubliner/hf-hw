import asyncio
import os
import re
import time
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_BASE = "https://huggingface.co/api"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Constants ---

GPUS = {
    "AMD": {
        "RX 7000 (RDNA 3)": [
            {"name": "RX 7600",     "vram": 8,  "bandwidth_gbs": 288},
            {"name": "RX 7600 XT",  "vram": 16, "bandwidth_gbs": 288},
            {"name": "RX 7700",     "vram": 12, "bandwidth_gbs": 432},
            {"name": "RX 7800 XT",  "vram": 16, "bandwidth_gbs": 624},
            {"name": "RX 7900 GRE", "vram": 16, "bandwidth_gbs": 576},
            {"name": "RX 7900 XT",  "vram": 20, "bandwidth_gbs": 800},
            {"name": "RX 7900 XTX", "vram": 24, "bandwidth_gbs": 960},
        ],
        "RX 9000 (RDNA 4)": [
            {"name": "RX 9070",    "vram": 16, "bandwidth_gbs": 672},
            {"name": "RX 9070 XT", "vram": 16, "bandwidth_gbs": 800},
        ],
    },
    "Nvidia": {
        "RTX 3000 (Ampere)": [
            {"name": "RTX 3060",     "vram": 12, "bandwidth_gbs": 360},
            {"name": "RTX 3060 Ti",  "vram": 8,  "bandwidth_gbs": 448},
            {"name": "RTX 3070",     "vram": 8,  "bandwidth_gbs": 448},
            {"name": "RTX 3070 Ti",  "vram": 8,  "bandwidth_gbs": 608},
            {"name": "RTX 3080 10G", "vram": 10, "bandwidth_gbs": 760},
            {"name": "RTX 3080 12G", "vram": 12, "bandwidth_gbs": 912},
            {"name": "RTX 3080 Ti",  "vram": 12, "bandwidth_gbs": 912},
            {"name": "RTX 3090",     "vram": 24, "bandwidth_gbs": 936},
            {"name": "RTX 3090 Ti",  "vram": 24, "bandwidth_gbs": 1008},
        ],
        "RTX 4000 (Ada)": [
            {"name": "RTX 4060",          "vram": 8,  "bandwidth_gbs": 272},
            {"name": "RTX 4060 Ti 8G",    "vram": 8,  "bandwidth_gbs": 288},
            {"name": "RTX 4060 Ti 16G",   "vram": 16, "bandwidth_gbs": 288},
            {"name": "RTX 4070",          "vram": 12, "bandwidth_gbs": 504},
            {"name": "RTX 4070 Super",    "vram": 12, "bandwidth_gbs": 504},
            {"name": "RTX 4070 Ti",       "vram": 12, "bandwidth_gbs": 504},
            {"name": "RTX 4070 Ti Super", "vram": 16, "bandwidth_gbs": 672},
            {"name": "RTX 4080",          "vram": 16, "bandwidth_gbs": 736},
            {"name": "RTX 4080 Super",    "vram": 16, "bandwidth_gbs": 736},
            {"name": "RTX 4090",          "vram": 24, "bandwidth_gbs": 1008},
        ],
        "RTX 5000 (Blackwell)": [
            {"name": "RTX 5070",    "vram": 12, "bandwidth_gbs": 672},
            {"name": "RTX 5070 Ti", "vram": 16, "bandwidth_gbs": 896},
            {"name": "RTX 5080",    "vram": 16, "bandwidth_gbs": 960},
            {"name": "RTX 5090",    "vram": 32, "bandwidth_gbs": 1792},
        ],
    },
    "Intel": {
        "Arc Alchemist": [
            {"name": "Arc A380", "vram": 6,  "bandwidth_gbs": 186},
            {"name": "Arc A580", "vram": 8,  "bandwidth_gbs": 512},
            {"name": "Arc A750", "vram": 8,  "bandwidth_gbs": 512},
            {"name": "Arc A770", "vram": 16, "bandwidth_gbs": 560},
        ],
        "Arc Battlemage": [
            {"name": "Arc B580", "vram": 12, "bandwidth_gbs": 456},
            {"name": "Arc B770", "vram": 16, "bandwidth_gbs": 672},
        ],
    },
}

QUANT_BITS = {
    "IQ1": ("1-bit", "red"),
    "IQ2": ("2-bit", "orange"),
    "IQ3": ("3-bit", "amber"),
    "Q2":  ("2-bit", "orange"),
    "Q3":  ("3-bit", "amber"),
    "Q4":  ("4-bit", "yellow"),
    "IQ4": ("4-bit", "yellow"),
    "TQ1": ("1-bit", "red"),
    "TQ2": ("2-bit", "orange"),
    "Q5":  ("5-bit", "lime"),
    "Q6":  ("6-bit", "green"),
    "Q8":  ("8-bit", "teal"),
    "F16": ("16-bit", "blue"),
    "BF16":("16-bit", "blue"),
}

SORT_MAP = {
    "trending":  "trendingScore",
    "downloads": "downloads",
    "likes":     "likes",
    "newest":    "createdAt",
    "updated":   "lastModified",
}

# --- Helpers ---

def quant_label(name: str) -> tuple[str, str]:
    upper = name.upper()
    for prefix, info in QUANT_BITS.items():
        if upper.startswith(prefix):
            return info
    return ("?-bit", "gray")

_QUANT_RE = re.compile(
    r"[_-]((?:IQ|TQ|BF?)\d[\w]*|Q\d[\w]*)(?:-\d{5}-of-\d{5})?\.gguf$",
    re.IGNORECASE,
)

MTP_PATTERN = re.compile(r"[_\-](mtp|draft|eagle|dflash)[_\-]", re.IGNORECASE)

def is_mtp_repo(model_id: str) -> bool:
    return bool(MTP_PATTERN.search(model_id))

def extract_base_model_id(tags: list[str]) -> str | None:
    for t in tags:
        if t.startswith("base_model:") and "quantized" not in t and "finetune" not in t:
            return t[len("base_model:"):]
    return None

# --- API Logic ---

async def fetch_tree(client: httpx.AsyncClient, model_id: str) -> list[dict]:
    url = f"{HF_BASE}/models/{model_id}/tree/main"
    try:
        resp = await client.get(
            url,
            params={"expand": "true", "limit": "100"},
            headers=HF_HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []

async def fetch_model_config(client: httpx.AsyncClient, base_id: str) -> dict:
    url = f"https://huggingface.co/{base_id}/resolve/main/config.json"
    try:
        resp = await client.get(url, headers=HF_HEADERS, timeout=10, follow_redirects=True)
        if resp.status_code != 200:
            return {}
        raw = resp.json()
    except Exception:
        return {}

    n_heads = raw.get("num_attention_heads") or 0
    hidden  = raw.get("hidden_size") or 0
    head_dim = (hidden // n_heads) if n_heads else 0

    return {
        "max_ctx": (raw.get("max_position_embeddings") or raw.get("seq_length") or raw.get("max_seq_len")),
        "n_layers":   raw.get("num_hidden_layers"),
        "n_kv_heads": raw.get("num_key_value_heads") or n_heads or None,
        "head_dim":   head_dim or None,
        "vocab_size": raw.get("vocab_size"),
    }

def has_gguf_siblings(model: dict) -> bool:
    siblings = model.get("siblings", [])
    if not model.get("library_name") and "gguf" in model.get("tags", []):
        return any(".gguf" in s.get("rfilename", "").lower() for s in siblings) or len(siblings) > 2
    return any(".gguf" in s.get("rfilename", "").lower() for s in siblings)

def group_quants(tree_entries: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for entry in tree_entries:
        path: str = entry.get("path", "")
        size: int = entry.get("size", 0)
        if not path.lower().endswith(".gguf"):
            continue
        basename = path.split("/")[-1]
        if (basename.lower().startswith(("mmproj", "vocab", "encoder"))
                or ".imatrix." in basename.lower()):
            continue

        m = _QUANT_RE.search(basename)
        quant_key = m.group(1).upper() if m else basename.replace(".gguf", "")

        if quant_key not in groups:
            groups[quant_key] = {"size_bytes": 0, "filename": path, "quant": quant_key}
        groups[quant_key]["size_bytes"] += size

    res = []
    for key, info in groups.items():
        label, color = quant_label(key)
        info["label"] = label
        info["color"] = color
        info["size_gb"] = round(info["size_bytes"] / 1e9, 2)
        res.append(info)
    return res

async def fetch_gguf_candidates(
    client: httpx.AsyncClient,
    sort: str,
    page: int,
    search: str,
    min_candidates: int = 100,
    max_hf_pages: int = 10,
) -> list[dict]:
    candidates: list[dict] = []
    seen_ids: set[str] = set()
    
    params: dict = {
        "tags": "gguf",
        "full": "true",
        "sort": SORT_MAP.get(sort, "trendingScore"),
        "direction": "-1",
        "limit": "100",
        "p": str(page),
        "search": search if search else "GGUF",
    }

    for p_idx in range(max_hf_pages):
        params["p"] = str(p_idx)
        print(f"Fetching page {p_idx}...")
        try:
            resp = await client.get(
                f"{HF_BASE}/models",
                params=params,
                headers=HF_HEADERS,
                timeout=20,
            )
            if resp.status_code != 200:
                break
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
                
            for m in batch:
                if m["id"] not in seen_ids and has_gguf_siblings(m):
                    candidates.append(m)
                    seen_ids.add(m["id"])
            
            if len(candidates) >= min_candidates:
                break
        except Exception as e:
            print(f"Error fetching page {p_idx}: {e}")
            break
            
    return candidates

# --- Main Logic ---

async def main():
    print("Starting update...")
    if not HF_TOKEN:
        print("Error: HF_TOKEN is not set.")
        sys.exit(1)

    async with httpx.AsyncClient() as client:
        models_data = await fetch_gguf_candidates(client, "trending", 0, "")
        print(f"Found {len(models_data)} candidate models. Processing...")

        results = []
        for i, m in enumerate(models_data):
            if i % 50 == 0:
                print(f"Processing model {i}/{len(models_data)}: {m['id']}")
            
            tree = await fetch_tree(client, m["id"])
            if not tree:
                continue
            
            base_id = extract_base_model_id(m.get("tags", []))
            cfg = await fetch_model_config(client, base_id) if base_id else {}

            quants = group_quants(tree)
            if not quants:
                continue

            results.append({
                "id": m["id"],
                "author": m.get("author", ""),
                "createdAt": m.get("createdAt", ""),
                "lastModified": m.get("lastModified", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "tags": [t for t in m.get("tags", []) if not any(t.startswith(p) for p in ("base_model:", "dataset:", "arxiv:", "region:", "license:"))][:8],
                "max_ctx": cfg.get("max_ctx"),
                "vocab_size": cfg.get("vocab_size"),
                "is_mtp": is_mtp_repo(m["id"]),
                "arch": {
                    "n_layers": cfg.get("n_layers"),
                    "n_kv_heads": cfg.get("n_kv_heads"),
                    "head_dim": cfg.get("head_dim")
                },
                "quants": quants
            })
        
        print(f"Writing {len(results)} models to data/models.json")
        os.makedirs("data", exist_ok=True)
        with open("data/models.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("Writing gpus.json")
        with open("data/gpus.json", "w") as f:
            json.dump(GPUS, f, indent=2)
        
        print("Writing metadata.json")
        with open("data/metadata.json", "w") as f:
            json.dump({"last_updated": datetime.utcnow().isoformat()}, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
