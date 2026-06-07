import asyncio
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_BASE = "https://huggingface.co/api"

app = FastAPI()

# --- GPU database ---

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

# --- Quant quality labels ---

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

def quant_label(name: str) -> tuple[str, str]:
    upper = name.upper()
    for prefix, info in QUANT_BITS.items():
        if upper.startswith(prefix):
            return info
    return ("?-bit", "gray")

# --- KV cache quantization ---

KV_QUANT_BPE: dict[str, float] = {
    "f16": 2.0,
    "q8":  1.0,
    "q6":  0.75,
    "q4":  0.5,
}

# --- In-memory caches (tree data + model configs) ---

_tree_cache: dict[str, tuple[float, list]] = {}
_config_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 3600

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


async def fetch_tree(client: httpx.AsyncClient, model_id: str) -> list[dict]:
    now = time.monotonic()
    if model_id in _tree_cache:
        ts, data = _tree_cache[model_id]
        if now - ts < CACHE_TTL:
            return data

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
        if not isinstance(data, list):
            return []
        _tree_cache[model_id] = (now, data)
        return data
    except Exception:
        return []


async def fetch_model_config(client: httpx.AsyncClient, base_id: str) -> dict:
    """Fetch config.json from a base model repo and extract architecture params."""
    now = time.monotonic()
    if base_id in _config_cache:
        ts, data = _config_cache[base_id]
        if now - ts < CACHE_TTL:
            return data

    url = f"https://huggingface.co/{base_id}/resolve/main/config.json"
    try:
        resp = await client.get(url, headers=HF_HEADERS, timeout=10, follow_redirects=True)
        if resp.status_code != 200:
            _config_cache[base_id] = (now, {})
            return {}
        raw = resp.json()
    except Exception:
        _config_cache[base_id] = (now, {})
        return {}

    n_heads = raw.get("num_attention_heads") or 0
    hidden  = raw.get("hidden_size") or 0
    head_dim = (hidden // n_heads) if n_heads else 0

    cfg = {
        "max_ctx": (
            raw.get("max_position_embeddings")
            or raw.get("seq_length")
            or raw.get("max_seq_len")
        ),
        "n_layers":   raw.get("num_hidden_layers"),
        "n_kv_heads": raw.get("num_key_value_heads") or n_heads or None,
        "head_dim":   head_dim or None,
        "vocab_size": raw.get("vocab_size"),
    }
    _config_cache[base_id] = (now, cfg)
    return cfg


def extract_base_model_id(tags: list[str]) -> str | None:
    for t in tags:
        if t.startswith("base_model:") and "quantized" not in t and "finetune" not in t:
            return t[len("base_model:"):]
    return None


# --- GGUF quant grouping ---

_QUANT_RE = re.compile(
    r"[_-]((?:IQ|TQ|BF?)\d[\w]*|Q\d[\w]*)(?:-\d{5}-of-\d{5})?\.gguf$",
    re.IGNORECASE,
)

MTP_PATTERN = re.compile(r"[_\-](mtp|draft|eagle|dflash)[_\-]", re.IGNORECASE)


def is_mtp_repo(model_id: str) -> bool:
    return bool(MTP_PATTERN.search(model_id))


def group_quants(tree_entries: list[dict]) -> dict[str, dict]:
    """Return {quant_key: {size_bytes, label, color, filename}} from tree entries."""
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

    for key, info in groups.items():
        label, color = quant_label(key)
        info["label"] = label
        info["color"] = color
        info["size_gb"] = round(info["size_bytes"] / 1e9, 2)

    return groups


def compute_kv_gb(cfg: dict, context_len: int, kv_quant: str) -> float | None:
    """Return KV cache VRAM in GB using architecture params, or None to use fallback."""
    n_layers   = cfg.get("n_layers")
    n_kv_heads = cfg.get("n_kv_heads")
    head_dim   = cfg.get("head_dim")
    if not (n_layers and n_kv_heads and head_dim):
        return None
    bpe = KV_QUANT_BPE.get(kv_quant, 2.0)
    return 2 * n_layers * n_kv_heads * head_dim * context_len * bpe / 1e9


def filter_quants(
    quants: dict[str, dict],
    available_vram_gb: float,
    context_len: int,
    kv_quant: str,
    cfg: dict,
) -> list[dict]:
    result = []
    for info in quants.values():
        sz = info["size_gb"]
        kv = compute_kv_gb(cfg, context_len, kv_quant)
        if kv is None:
            # Proportional fallback when architecture is unknown
            kv = sz * 0.1 * (context_len / 8192)
        kv = round(kv, 2)
        if sz + kv <= available_vram_gb * 0.95:
            result.append({**info, "kv_cache_gb": kv})
    result.sort(key=lambda x: x["size_bytes"])
    return result


# --- HF API helpers ---

SORT_MAP = {
    "trending":  "trendingScore",
    "downloads": "downloads",
    "likes":     "likes",
    "newest":    "createdAt",
    "updated":   "lastModified",
}


async def fetch_model_list(
    client: httpx.AsyncClient,
    sort: str,
    page: int,
    search: str,
) -> list[dict]:
    params: dict = {
        "tags": "gguf",
        "full": "true",
        "sort": SORT_MAP.get(sort, "trendingScore"),
        "direction": "-1",
        "limit": "100",
        "p": str(page),
        "search": search if search else "GGUF",
    }
    resp = await client.get(
        f"{HF_BASE}/models",
        params=params,
        headers=HF_HEADERS,
        timeout=20,
    )
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data if isinstance(data, list) else []


def has_gguf_siblings(model: dict) -> bool:
    siblings = model.get("siblings", [])
    if not model.get("library_name") and "gguf" in model.get("tags", []):
        return any(".gguf" in s.get("rfilename", "").lower() for s in siblings) or len(siblings) > 2
    return any(".gguf" in s.get("rfilename", "").lower() for s in siblings)


async def fetch_gguf_candidates(
    client: httpx.AsyncClient,
    sort: str,
    page: int,
    search: str,
    min_candidates: int = 20,
    max_hf_pages: int = 5,
) -> list[dict]:
    candidates: list[dict] = []
    seen_ids: set[str] = set()
    hf_page = page * max_hf_pages
    for _ in range(max_hf_pages):
        batch = await fetch_model_list(client, sort, hf_page, search)
        if not batch:
            break
        for m in batch:
            if m["id"] not in seen_ids and has_gguf_siblings(m):
                candidates.append(m)
                seen_ids.add(m["id"])
        hf_page += 1
        if len(candidates) >= min_candidates:
            break
    return candidates[:25]


# --- API routes ---

@app.get("/api/gpus")
def get_gpus():
    return GPUS


@app.get("/api/models")
async def get_models(
    vram_gb:   float = Query(24.0),
    context:   int   = Query(8192),
    kv_quant:  str   = Query("f16"),
    sort:      str   = Query("trending"),
    page:      int   = Query(0),
    search:    str   = Query(""),
):
    async with httpx.AsyncClient() as client:
        gguf_models = await fetch_gguf_candidates(client, sort, page, search)

        async def _no_config():
            return {}

        # Fetch tree data + base model configs concurrently
        trees, configs = await asyncio.gather(
            asyncio.gather(*[fetch_tree(client, m["id"]) for m in gguf_models]),
            asyncio.gather(*[
                fetch_model_config(client, bid)
                if (bid := extract_base_model_id(m.get("tags", []))) else _no_config()
                for m in gguf_models
            ]),
        )

    results = []
    for model, tree, cfg in zip(gguf_models, trees, configs):
        quants = group_quants(tree)
        compatible = filter_quants(quants, vram_gb, context, kv_quant, cfg)
        if not compatible:
            continue

        results.append({
            "id":           model["id"],
            "author":       model.get("author", ""),
            "createdAt":    model.get("createdAt", ""),
            "lastModified": model.get("lastModified", ""),
            "downloads":    model.get("downloads", 0),
            "likes":        model.get("likes", 0),
            "pipeline_tag": model.get("pipeline_tag", ""),
            "tags": [
                t for t in model.get("tags", [])
                if not any(t.startswith(p) for p in
                           ("base_model:", "dataset:", "arxiv:", "region:", "license:"))
            ][:8],
            "quants":    compatible,
            "max_ctx":   cfg.get("max_ctx"),
            "vocab_size":cfg.get("vocab_size"),
            "is_mtp":    is_mtp_repo(model["id"]),
        })

    return results


# --- Static files ---

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
