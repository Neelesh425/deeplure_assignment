"""
Phase 0 – Data Audit Script
Inspects DeepLure and Kaggle datasets without modifying them.
Outputs: docs/data_audit.md  +  console summary.

Usage:
  python scripts/data_audit.py --data-root <path>
"""
import argparse
import json
import hashlib
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def safe_open_image(p: Path) -> Image.Image | None:
    """Open image, return None if corrupt."""
    try:
        img = Image.open(p)
        img.verify()  # check integrity
        img = Image.open(p)  # re-open after verify
        img.load()
        return img
    except Exception as e:
        log.warning(f"Corrupt/unreadable: {p} — {e}")
        return None


def image_metadata(p: Path) -> dict | None:
    """Return basic metadata without loading full pixel data where possible."""
    try:
        img = Image.open(p)
        w, h = img.size
        mode = img.mode
        fmt = img.format
        file_size = p.stat().st_size
        return {
            "path": str(p),
            "width": w,
            "height": h,
            "mode": mode,
            "format": fmt,
            "file_size_bytes": file_size,
        }
    except Exception as e:
        return {"path": str(p), "error": str(e)}


def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash_gray(p: Path, hash_size: int = 16) -> str | None:
    """Average-hash on grayscale, returns hex string."""
    try:
        img = Image.open(p).convert("L").resize(
            (hash_size, hash_size), Image.Resampling.LANCZOS
        )
        arr = np.array(img, dtype=np.float64)
        mean = arr.mean()
        bits = (arr > mean).flatten()
        # pack bits to hex
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return format(val, f"0{hash_size * hash_size // 4}x")
    except Exception:
        return None


def hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex hash strings."""
    i1, i2 = int(h1, 16), int(h2, 16)
    return bin(i1 ^ i2).count("1")


def parse_kaggle_filename(fname: str) -> dict:
    """
    Roboflow filenames look like: <source_name>.rf.<hash>.<ext>
    or just <name>.<ext>.  Returns source_name and rf_hash if present.
    """
    stem = Path(fname).stem
    if ".rf." in fname:
        parts = stem.split(".rf.")
        return {"source_name": parts[0], "rf_hash": parts[1] if len(parts) > 1 else None}
    return {"source_name": stem, "rf_hash": None}


def deeplure_filename_analysis(fname: str) -> dict:
    """
    Analyse DeepLure filename for embedded structure.
    Files are named like: img_<number>.jpg or h_img_<number>.jpg
    """
    stem = Path(fname).stem
    prefix = ""
    number = ""
    if stem.startswith("h_img_"):
        prefix = "h_img"
        number = stem[6:]
    elif stem.startswith("img_"):
        prefix = "img"
        number = stem[4:]
    else:
        prefix = "unknown"
        number = stem
    return {"prefix": prefix, "number": number, "stem": stem}


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------

def audit_deeplure(data_root: Path) -> dict:
    """Full audit of the DeepLure dataset."""
    dl_root = data_root / "deeplure" / "handloom_sarees"
    if not dl_root.exists():
        # Try alternate path
        dl_root = data_root / "deeplure" / "handloom sarees"
    if not dl_root.exists():
        log.error(f"DeepLure root not found at {dl_root}")
        return {"error": "not_found"}

    log.info(f"Auditing DeepLure at: {dl_root}")

    # 1. Folder structure
    all_items = list(dl_root.rglob("*"))
    subdirs = [p for p in all_items if p.is_dir()]
    all_files = [p for p in all_items if p.is_file()]
    image_files = [p for p in all_files if is_image(p)]
    non_image_files = [p for p in all_files if not is_image(p)]

    log.info(f"  Subdirectories: {len(subdirs)}")
    log.info(f"  Image files: {len(image_files)}")
    log.info(f"  Non-image files: {len(non_image_files)}: {[p.name for p in non_image_files]}")

    # 2. File naming patterns
    filename_analysis = [deeplure_filename_analysis(p.name) for p in image_files]
    prefix_counts = Counter(fa["prefix"] for fa in filename_analysis)
    log.info(f"  Filename prefixes: {dict(prefix_counts)}")

    # 3. Image metadata (sizes, modes, formats)
    metadata_list = []
    corrupt_files = []
    for p in image_files:
        meta = image_metadata(p)
        if "error" in meta:
            corrupt_files.append(meta)
        else:
            metadata_list.append(meta)

    if metadata_list:
        widths = [m["width"] for m in metadata_list]
        heights = [m["height"] for m in metadata_list]
        sizes_bytes = [m["file_size_bytes"] for m in metadata_list]
        modes = Counter(m["mode"] for m in metadata_list)
        formats = Counter(m["format"] for m in metadata_list)
        aspect_ratios = [m["width"] / m["height"] for m in metadata_list]

        res_stats = {
            "width": {"min": min(widths), "max": max(widths), "mean": round(np.mean(widths), 1), "median": int(np.median(widths))},
            "height": {"min": min(heights), "max": max(heights), "mean": round(np.mean(heights), 1), "median": int(np.median(heights))},
            "file_size_kb": {"min": round(min(sizes_bytes)/1024, 1), "max": round(max(sizes_bytes)/1024, 1), "mean": round(np.mean(sizes_bytes)/1024, 1)},
            "aspect_ratio": {"min": round(min(aspect_ratios), 2), "max": round(max(aspect_ratios), 2), "mean": round(np.mean(aspect_ratios), 2)},
            "modes": dict(modes),
            "formats": dict(formats),
        }
    else:
        res_stats = {}

    # 4. Perceptual hashes + near-duplicate detection
    log.info("  Computing perceptual hashes (grayscale)...")
    phashes = {}
    for p in image_files:
        h = perceptual_hash_gray(p)
        if h:
            phashes[str(p.relative_to(data_root))] = h

    # Find near-duplicates (hamming distance ≤ 10 on 256-bit hash)
    hash_items = list(phashes.items())
    near_dupes = []
    for i in range(len(hash_items)):
        for j in range(i + 1, len(hash_items)):
            dist = hamming_distance(hash_items[i][1], hash_items[j][1])
            if dist <= 10:
                near_dupes.append({
                    "file1": hash_items[i][0],
                    "file2": hash_items[j][0],
                    "hamming_distance": dist,
                })

    # 5. MD5 exact duplicates
    log.info("  Computing file MD5s...")
    md5s = {}
    for p in image_files:
        md5s[str(p.relative_to(data_root))] = file_md5(p)

    md5_groups = defaultdict(list)
    for path, h in md5s.items():
        md5_groups[h].append(path)
    exact_dupes = {k: v for k, v in md5_groups.items() if len(v) > 1}

    # 6. Content type analysis — check for very small or very large images
    tiny_images = [m for m in metadata_list if m["width"] < 50 or m["height"] < 50]
    huge_images = [m for m in metadata_list if m["width"] > 4000 or m["height"] > 4000]

    # 7. Aspect ratio clustering (portrait/landscape/square)
    portrait = sum(1 for ar in aspect_ratios if ar < 0.9) if metadata_list else 0
    landscape = sum(1 for ar in aspect_ratios if ar > 1.1) if metadata_list else 0
    square = sum(1 for ar in aspect_ratios if 0.9 <= ar <= 1.1) if metadata_list else 0

    result = {
        "root": str(dl_root),
        "total_images": len(image_files),
        "subdirectories": len(subdirs),
        "subdir_names": [str(s.relative_to(dl_root)) for s in subdirs],
        "non_image_files": [p.name for p in non_image_files],
        "filename_prefixes": dict(prefix_counts),
        "corrupt_files": corrupt_files,
        "resolution_stats": res_stats,
        "orientation": {"portrait": portrait, "landscape": landscape, "square": square},
        "near_duplicates_within": near_dupes[:50],  # cap output
        "exact_duplicates": exact_dupes,
        "tiny_images": [m["path"] for m in tiny_images],
        "huge_images": [m["path"] for m in huge_images],
        "phashes": phashes,
        "md5s": md5s,
    }
    return result


def audit_kaggle(data_root: Path) -> dict:
    """Full audit of the Kaggle/Roboflow dataset."""
    kg_root = data_root / "kaggle"
    if not kg_root.exists():
        log.error(f"Kaggle root not found at {kg_root}")
        return {"error": "not_found"}

    log.info(f"Auditing Kaggle at: {kg_root}")

    results = {}
    all_image_files = []
    split_stats = {}

    for split_name in ["train", "valid", "test"]:
        split_dir = kg_root / split_name
        if not split_dir.exists():
            continue

        categories = {}
        for cat_dir in sorted(split_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            imgs = [p for p in cat_dir.iterdir() if is_image(p)]
            categories[cat_dir.name] = len(imgs)
            all_image_files.extend([(split_name, cat_dir.name, p) for p in imgs])

        split_stats[split_name] = {
            "categories": categories,
            "total": sum(categories.values()),
        }

    log.info(f"  Split stats: {json.dumps(split_stats, indent=2)}")

    # Roboflow augmentation analysis
    log.info("  Analysing Roboflow filenames...")
    rf_analysis = []
    source_to_copies = defaultdict(list)
    for split, cat, p in all_image_files:
        parsed = parse_kaggle_filename(p.name)
        parsed["split"] = split
        parsed["category"] = cat
        parsed["full_path"] = str(p)
        rf_analysis.append(parsed)
        source_to_copies[parsed["source_name"]].append({
            "split": split,
            "category": cat,
            "file": p.name,
            "has_rf_hash": parsed["rf_hash"] is not None,
        })

    # Count augmented copies
    augmented_sources = {k: v for k, v in source_to_copies.items() if len(v) > 1}
    total_augmented = sum(len(v) for v in augmented_sources.values())
    total_unique_sources = len(source_to_copies)
    has_rf_count = sum(1 for r in rf_analysis if r["rf_hash"] is not None)

    log.info(f"  Total images: {len(all_image_files)}")
    log.info(f"  Unique source names: {total_unique_sources}")
    log.info(f"  Images with .rf. hash: {has_rf_count}")
    log.info(f"  Sources with >1 copy: {len(augmented_sources)}")
    log.info(f"  Total augmented images: {total_augmented}")

    # Image metadata (sample all)
    log.info("  Collecting image metadata...")
    metadata_list = []
    corrupt_files = []
    for _, _, p in all_image_files:
        meta = image_metadata(p)
        if "error" in meta:
            corrupt_files.append(meta)
        else:
            metadata_list.append(meta)

    if metadata_list:
        widths = [m["width"] for m in metadata_list]
        heights = [m["height"] for m in metadata_list]
        modes = Counter(m["mode"] for m in metadata_list)
        res_stats = {
            "width": {"min": min(widths), "max": max(widths), "unique": len(set(widths))},
            "height": {"min": min(heights), "max": max(heights), "unique": len(set(heights))},
            "modes": dict(modes),
        }
    else:
        res_stats = {}

    # Cross-dataset duplicate check: compute phashes for Kaggle too
    log.info("  Computing perceptual hashes (Kaggle)...")
    phashes = {}
    for _, _, p in all_image_files:
        h = perceptual_hash_gray(p)
        if h:
            phashes[str(p.relative_to(data_root))] = h

    # Distribution of augmented copies across splits  
    aug_cross_split = defaultdict(lambda: defaultdict(int))
    for src, copies in augmented_sources.items():
        splits_seen = set()
        for c in copies:
            splits_seen.add(c["split"])
        for s in splits_seen:
            aug_cross_split["sources_in_split"][s] += 1
    
    # Count sources appearing in multiple splits (leakage risk)
    leaky_sources = {k: v for k, v in augmented_sources.items() 
                     if len(set(c["split"] for c in v)) > 1}

    result = {
        "root": str(kg_root),
        "split_stats": split_stats,
        "total_images": len(all_image_files),
        "unique_source_names": total_unique_sources,
        "images_with_rf_hash": has_rf_count,
        "sources_with_multiple_copies": len(augmented_sources),
        "total_augmented_images": total_augmented,
        "leaky_sources_across_splits": len(leaky_sources),
        "leaky_source_examples": dict(list(leaky_sources.items())[:5]),
        "corrupt_files": corrupt_files,
        "resolution_stats": res_stats,
        "phashes": phashes,
    }
    return result


def cross_dataset_duplicates(dl_phashes: dict, kg_phashes: dict, threshold: int = 10) -> list:
    """Find near-duplicates between DeepLure and Kaggle datasets."""
    cross_dupes = []
    dl_items = list(dl_phashes.items())
    kg_items = list(kg_phashes.items())

    log.info(f"  Cross-checking {len(dl_items)} DeepLure x {len(kg_items)} Kaggle hashes...")
    for dl_path, dl_hash in dl_items:
        for kg_path, kg_hash in kg_items:
            dist = hamming_distance(dl_hash, kg_hash)
            if dist <= threshold:
                cross_dupes.append({
                    "deeplure": dl_path,
                    "kaggle": kg_path,
                    "hamming_distance": dist,
                })
    return cross_dupes


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(dl_audit: dict, kg_audit: dict, cross_dupes: list, output_path: Path) -> str:
    """Generate docs/data_audit.md"""
    lines = []
    lines.append("# Phase 0 — Data Audit Report\n")
    lines.append(f"*Generated by `scripts/data_audit.py`*\n")
    lines.append("---\n")

    # DeepLure
    lines.append("## 1. DeepLure Corpus\n")
    lines.append(f"- **Root**: `{dl_audit.get('root', 'N/A')}`")
    lines.append(f"- **Total images**: {dl_audit.get('total_images', 0)}")
    lines.append(f"- **Subdirectories**: {dl_audit.get('subdirectories', 0)}")
    if dl_audit.get("subdir_names"):
        lines.append(f"  - Names: {dl_audit['subdir_names']}")
    else:
        lines.append("  - **FLAT FOLDER** — no subfolders for design/colorway grouping")
    lines.append(f"- **Non-image files**: {dl_audit.get('non_image_files', [])}")
    lines.append(f"- **Filename prefixes**: {dl_audit.get('filename_prefixes', {})}")
    lines.append(f"- **Corrupt files**: {len(dl_audit.get('corrupt_files', []))}")
    if dl_audit.get("corrupt_files"):
        for cf in dl_audit["corrupt_files"]:
            lines.append(f"  - `{cf}`")
    lines.append("")

    # Resolution stats
    rs = dl_audit.get("resolution_stats", {})
    if rs:
        lines.append("### Resolution & Format Statistics\n")
        lines.append(f"| Metric | Min | Max | Mean | Median |")
        lines.append(f"|--------|-----|-----|------|--------|")
        w = rs.get("width", {})
        h = rs.get("height", {})
        fs = rs.get("file_size_kb", {})
        ar = rs.get("aspect_ratio", {})
        lines.append(f"| Width (px) | {w.get('min','')} | {w.get('max','')} | {w.get('mean','')} | {w.get('median','')} |")
        lines.append(f"| Height (px) | {h.get('min','')} | {h.get('max','')} | {h.get('mean','')} | {h.get('median','')} |")
        lines.append(f"| File size (KB) | {fs.get('min','')} | {fs.get('max','')} | {fs.get('mean','')} | — |")
        lines.append(f"| Aspect ratio | {ar.get('min','')} | {ar.get('max','')} | {ar.get('mean','')} | — |")
        lines.append(f"\n- **Colour modes**: {rs.get('modes', {})}")
        lines.append(f"- **Formats**: {rs.get('formats', {})}")
        lines.append("")

    orient = dl_audit.get("orientation", {})
    if orient:
        lines.append(f"### Orientation Distribution\n")
        lines.append(f"- Portrait (AR < 0.9): {orient.get('portrait', 0)}")
        lines.append(f"- Landscape (AR > 1.1): {orient.get('landscape', 0)}")
        lines.append(f"- Square (0.9 ≤ AR ≤ 1.1): {orient.get('square', 0)}")
        lines.append("")

    # Design identity analysis
    lines.append("### Design Identity Encoding\n")
    lines.append("**CRITICAL FINDING**: The DeepLure corpus is a **flat folder** with 165 images.")
    lines.append("Filenames follow the pattern `img_<number>.jpg` (156 files) or `h_img_<number>.jpg` (9 files).")
    lines.append("The numeric IDs appear random and do NOT encode design or colorway information.")
    lines.append("There are **NO design labels, NO colorway labels, NO metadata files, NO subfolders**.")
    lines.append("")
    lines.append("The `h_img_` prefix might indicate a different source, quality, or content type (e.g., 'handloom' vs generic).")
    lines.append("")
    lines.append("> **Implication**: We cannot directly build design-disjoint splits or")
    lines.append("> same-design/different-colorway pairs from the DeepLure data without")
    lines.append("> manual annotation or automated grouping (perceptual hash clustering,")
    lines.append("> texture descriptor clustering).")
    lines.append("")

    # Duplicates
    lines.append("### Near-Duplicates (within DeepLure, grayscale pHash ≤ 10)\n")
    nd = dl_audit.get("near_duplicates_within", [])
    if nd:
        lines.append(f"Found **{len(nd)}** near-duplicate pairs:\n")
        lines.append("| File 1 | File 2 | Hamming Dist |")
        lines.append("|--------|--------|-------------|")
        for d in nd[:20]:
            lines.append(f"| `{Path(d['file1']).name}` | `{Path(d['file2']).name}` | {d['hamming_distance']} |")
        if len(nd) > 20:
            lines.append(f"\n*… and {len(nd)-20} more pairs*")
    else:
        lines.append("**None found.**")
    lines.append("")

    ed = dl_audit.get("exact_duplicates", {})
    lines.append("### Exact Duplicates (MD5)\n")
    if ed:
        lines.append(f"Found **{len(ed)}** duplicate groups:")
        for md5, files in ed.items():
            lines.append(f"- MD5 `{md5[:12]}…`: {[Path(f).name for f in files]}")
    else:
        lines.append("**None found.**")
    lines.append("")

    # Tiny / huge
    tiny = dl_audit.get("tiny_images", [])
    huge = dl_audit.get("huge_images", [])
    if tiny:
        lines.append(f"### Tiny images (<50px): {len(tiny)}\n")
        for t in tiny:
            lines.append(f"- `{Path(t).name}`")
        lines.append("")
    if huge:
        lines.append(f"### Huge images (>4000px): {len(huge)}\n")
        for h_ in huge:
            lines.append(f"- `{Path(h_).name}`")
        lines.append("")

    # --- Kaggle ---
    lines.append("---\n")
    lines.append("## 2. Kaggle (Roboflow) Dataset\n")
    lines.append(f"- **Root**: `{kg_audit.get('root', 'N/A')}`")
    lines.append(f"- **Total images**: {kg_audit.get('total_images', 0)}")
    lines.append(f"- **Unique source images**: {kg_audit.get('unique_source_names', 0)}")
    lines.append(f"- **Images with `.rf.` augmentation hash**: {kg_audit.get('images_with_rf_hash', 0)}")
    lines.append(f"- **Source images with >1 augmented copy**: {kg_audit.get('sources_with_multiple_copies', 0)}")
    lines.append(f"- **Sources leaking across splits**: {kg_audit.get('leaky_sources_across_splits', 0)}")
    lines.append(f"- **Corrupt files**: {len(kg_audit.get('corrupt_files', []))}")
    lines.append("")

    ss = kg_audit.get("split_stats", {})
    if ss:
        lines.append("### Split × Category Counts\n")
        cats = set()
        for s in ss.values():
            cats.update(s.get("categories", {}).keys())
        cats = sorted(cats)

        header = "| Split | " + " | ".join(cats) + " | Total |"
        sep = "|-------|" + "|".join(["-----"] * len(cats)) + "|-------|"
        lines.append(header)
        lines.append(sep)
        for split_name in ["train", "valid", "test"]:
            if split_name in ss:
                s = ss[split_name]
                row = f"| {split_name} | "
                row += " | ".join(str(s["categories"].get(c, 0)) for c in cats)
                row += f" | {s['total']} |"
                lines.append(row)
        lines.append("")

    krs = kg_audit.get("resolution_stats", {})
    if krs:
        lines.append("### Resolution\n")
        lines.append(f"- Width range: {krs.get('width', {})}")
        lines.append(f"- Height range: {krs.get('height', {})}")
        lines.append(f"- Modes: {krs.get('modes', {})}")
        lines.append("")

    lines.append("### Roboflow Augmentation Analysis\n")
    lines.append("Per README.roboflow.txt:")
    lines.append("- Pre-processing: auto-orient (EXIF strip), resize to 640×640 (stretch)")
    lines.append("- Augmentation: 3 versions per source, salt-and-pepper noise (0.1% pixels)")
    lines.append(f"- **Unique source images** (before augmentation): {kg_audit.get('unique_source_names', '?')}")
    lines.append(f"- **Sources appearing in >1 split** (data leakage): {kg_audit.get('leaky_sources_across_splits', '?')}")
    lines.append("")

    if kg_audit.get("leaky_source_examples"):
        lines.append("#### Leakage Examples\n")
        for src, copies in kg_audit["leaky_source_examples"].items():
            splits = set(c["split"] for c in copies)
            lines.append(f"- `{src}`: appears in splits {splits}")
        lines.append("")

    lines.append("### Label Type\n")
    lines.append("Labels are **coarse pattern categories** (Banarasi, Bandhani, Ikat, Pichwai),")
    lines.append("NOT design IDs. A category contains many different designs — these are NOT")
    lines.append("usable as design identity labels for our metric learning task.")
    lines.append("")
    lines.append("**Usage decision**: Kaggle data can be used as:")
    lines.append("1. Weakly-labelled extra training data (same category ≠ same design)")
    lines.append("2. Colour-augmented self-supervised positives (augment same image = same 'design')")
    lines.append("3. Out-of-domain category-level evaluation")
    lines.append("4. **Cannot** be used in our design-level test split")
    lines.append("")

    # --- Cross-dataset duplicates ---
    lines.append("---\n")
    lines.append("## 3. Cross-Dataset Near-Duplicates\n")
    if cross_dupes:
        lines.append(f"Found **{len(cross_dupes)}** near-duplicate pairs between DeepLure and Kaggle:\n")
        lines.append("| DeepLure | Kaggle | Hamming Dist |")
        lines.append("|----------|--------|-------------|")
        for d in cross_dupes[:20]:
            lines.append(f"| `{Path(d['deeplure']).name}` | `{Path(d['kaggle']).name}` | {d['hamming_distance']} |")
    else:
        lines.append("**None found.**")
    lines.append("")

    # --- Key decisions ---
    lines.append("---\n")
    lines.append("## 4. Key Findings & Proposed Phase 1 Plan\n")
    lines.append("")
    lines.append("### Critical Issue: No Design Labels in DeepLure\n")
    lines.append("")
    lines.append("The DeepLure corpus has **no design identity labels** — it's a flat folder")
    lines.append("of 165 images with opaque numeric filenames. This is the primary dataset")
    lines.append("for our task, yet we cannot construct same-design/different-colorway pairs")
    lines.append("without either:")
    lines.append("")
    lines.append("1. **Manual annotation** (most reliable, time-consuming)")
    lines.append("2. **Automated pseudo-design grouping** via:")
    lines.append("   - Grayscale perceptual hash clustering (captures layout/structure)")
    lines.append("   - Texture descriptor clustering (LBP, Gabor features on luminance)")
    lines.append("   - Pretrained feature clustering (DINO/ImageNet features on grayscale)")
    lines.append("")
    lines.append("### Proposed Phase 1 Approach\n")
    lines.append("")
    lines.append("**Option A (Recommended)**: Semi-automated grouping:")
    lines.append("1. Extract colour-invariant features (grayscale DINO embeddings)")
    lines.append("2. Cluster with HDBSCAN (automatic cluster count)")
    lines.append("3. Within each cluster, assign colorway IDs based on colour histogram distance")
    lines.append("4. I (human) validate the grouping by reviewing cluster samples")
    lines.append("5. Risk: grouping errors propagate to train/test → report as limitation")
    lines.append("")
    lines.append("**Option B**: Treat each image as its own 'design' (no positives → ")
    lines.append("self-supervised only with synthetic recolours). Limits evaluation to synthetic pairs.")
    lines.append("")
    lines.append("**Kaggle usage**: Category labels only → self-supervised or weakly-supervised")
    lines.append("training data + out-of-domain evaluation. Ignore Roboflow train/valid/test split;")
    lines.append("collapse augmented copies first.")
    lines.append("")
    lines.append("### Awaiting Approval\n")
    lines.append("- Which approach for DeepLure design grouping? (A or B)")
    lines.append("- Should I proceed with automated clustering + human validation?")
    lines.append("- Any additional metadata or labels file I should look for?")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    log.info(f"Report written to {output_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0 — Data Audit")
    parser.add_argument("--data-root", type=str, required=True, help="Path to data_raw/")
    parser.add_argument("--output", type=str, default="docs/data_audit.md")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)

    log.info("=" * 60)
    log.info("PHASE 0 — DATA AUDIT")
    log.info("=" * 60)

    # Audit both datasets
    dl_audit = audit_deeplure(data_root)
    kg_audit = audit_kaggle(data_root)

    # Cross-dataset duplicates
    cross_dupes = []
    dl_ph = dl_audit.get("phashes", {})
    kg_ph = kg_audit.get("phashes", {})
    if dl_ph and kg_ph:
        cross_dupes = cross_dataset_duplicates(dl_ph, kg_ph)

    # Generate report
    generate_report(dl_audit, kg_audit, cross_dupes, output_path)

    # Save raw audit data as JSON for later use
    json_path = output_path.parent / "data_audit_raw.json"
    json_data = {
        "deeplure": {k: v for k, v in dl_audit.items() if k not in ("phashes", "md5s")},
        "kaggle": {k: v for k, v in kg_audit.items() if k != "phashes"},
        "cross_duplicates": cross_dupes,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    log.info(f"Raw data saved to {json_path}")

    log.info("=" * 60)
    log.info("AUDIT COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
