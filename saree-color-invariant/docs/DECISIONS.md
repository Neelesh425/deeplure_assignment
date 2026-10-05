# Decision Log

Running record of key decisions for viva preparation. Format: Decision → Alternatives → Reason → Evidence.

---

## Phase 0 — Data Audit

### D001: DeepLure has no design labels
- **What**: DeepLure corpus is a flat folder of 165 images with opaque numeric filenames. No subfolders, no metadata, no design/colorway labels.
- **Alternatives considered**: (1) Assume filenames encode something (they don't — numbers appear random). (2) Look for a separate CSV/JSON metadata file (none exists).
- **Reason**: This is a hard constraint of the data. Must either annotate manually or cluster automatically.
- **Evidence**: `scripts/data_audit.py` output — 0 subdirectories, 165 files, all named `img_<n>.jpg` or `h_img_<n>.jpg`.

### D002: Kaggle labels are category-level, not design-level
- **What**: Kaggle set has 4 coarse classes (Banarasi, Bandhani, Ikat, Pichwai). These are style categories containing many different designs.
- **Alternatives**: (1) Treat each category as a "design" — WRONG, conflates different designs. (2) Use as-is for metric learning — would learn category retrieval, not design retrieval.
- **Reason**: Using category labels as design IDs would make the model learn the wrong task. Each category has ~100-400 images of different designs.
- **Evidence**: Audit shows 499 unique source images across 4 categories. Same category ≠ same design.

### D003: Kaggle Roboflow split has leakage
- **What**: 31 source images appear in multiple splits (augmented copies leak across train/valid/test).
- **Alternatives**: (1) Use Roboflow splits as-is — WRONG, leakage. (2) Ignore their splits entirely (chosen).
- **Reason**: Data leakage invalidates evaluation. We collapse augmented copies and re-split ourselves.
- **Evidence**: Audit found 31 leaky source names (e.g., `image21_jpeg` in both train and valid).

### D004: `h_img_` prefix — separate handling
- **What**: 9 files have `h_img_` prefix, 156 have `img_`. The `h_img_` files tend to be larger (730–1512px).
- **Alternatives**: (1) Treat all the same (chosen for now). (2) Separate as different data source.
- **Reason**: Without documentation, the prefix meaning is unknown. Both sets are all-RGB JPEG, nearly-square. Treat uniformly but document the distinction.
- **Evidence**: Resolution analysis shows h_img_ range 610-1512px, img_ range 226-1512px. Both prefixes have similar aspect ratios (~1.0).

### D005: Proposed design grouping approach
- **What**: Semi-automated clustering (Option A) — grayscale DINO embeddings → HDBSCAN → human validation.
- **Alternatives**: (1) Manual annotation of all 165 (reliable but time-consuming). (2) Each image = own design (Option B — limits eval to synthetic pairs only). (3) Perceptual hash clustering (too coarse for design similarity).
- **Reason**: 165 images is manageable for human validation. DINO features on grayscale capture structural/texture similarity independent of colour. HDBSCAN doesn't need k. Human review catches errors.
- **Risk**: Clustering errors propagate. Mitigated by human review + reporting as limitation.
- **Awaiting**: User approval on approach choice.
