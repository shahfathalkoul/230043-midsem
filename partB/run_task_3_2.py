import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── 0. Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs("results", exist_ok=True)
print(f"Random seed: {RANDOM_SEED} | results/ directory ready\n")

# ── 1. Interestingness formula (Section 2) ────────────────────────────────────
def calculate_interestingness(count_product, count_total, exponent=0.95):
    if count_total <= 0:
        return 0.0
    return count_product / (count_total ** exponent)

def calculate_interestingness_smoothed(count_product, count_total, exponent=0.95, smoothing_k=1.0):
    if (count_total + smoothing_k) <= 0:
        return 0.0
    return count_product / ((count_total + smoothing_k) ** exponent)

# Sanity check
print("=== Section 2 Formula — Sanity Check ===")
examples = [
    ("baby shaker", 80,   100),
    ("lost phone",  50,  5000),
    ("iPod",       200,   300),
    ("software",    10, 10000),
]
print(f"{'Phrase':<20} {'count_p,e':>10} {'count_total':>12} {'score (a=0.95)':>16}")
print("-" * 62)
for phrase, cp, ct in examples:
    score = calculate_interestingness(cp, ct)
    print(f"{phrase:<20} {cp:>10} {ct:>12} {score:>16.4f}")

# ── 2. Toy dataset ────────────────────────────────────────────────────────────
ipod_phrases = [
    "baby shaker", "lost phone", "ipod touch", "itunes sync",
    "earbuds broke", "click wheel", "nano scratch", "apple store",
    "ipod shuffle", "music library", "album art", "podcast app",
    "battery drain", "firmware update", "ipod classic", "ios update",
    "genius playlist", "airplay mirror", "lightning cable", "dock connector",
]
it_phrases = [
    "software update", "data loss", "network outage", "cloud storage",
    "cyber attack", "server crash", "open source", "machine learning",
    "api integration", "devops pipeline", "agile sprint", "tech layoffs",
    "startup funding", "saas platform", "data breach", "zero day exploit",
    "blockchain adoption", "remote work", "digital transformation", "ai ethics",
    "microservices", "kubernetes deploy", "sql injection", "latency issue",
    "bandwidth limit", "edge computing", "iot device", "5g rollout",
    "encryption key", "password manager",
]

rows = []
for i in range(60):
    phrase = ipod_phrases[i % len(ipod_phrases)]
    if phrase in ("baby shaker", "lost phone"):
        cp = int(np.random.randint(60, 100))
        ct = int(np.random.randint(80, 200))
    else:
        cp = int(np.random.randint(5, 60))
        ct = int(np.random.randint(cp, cp * np.random.randint(5, 30)))
    rows.append({"phrase": phrase, "corpus": "iPod", "count_product": cp, "count_total": ct})

for i in range(50):
    phrase = it_phrases[i % len(it_phrases)]
    ct = int(np.random.randint(1000, 20000))
    cp = int(np.random.randint(1, max(2, ct // 50)))
    rows.append({"phrase": phrase, "corpus": "IT Industry", "count_product": cp, "count_total": ct})

df = pd.DataFrame(rows)
df["interestingness"] = df.apply(
    lambda r: calculate_interestingness(r["count_product"], r["count_total"]), axis=1)

print(f"\n=== Dataset ===")
print(f"Shape: {df.shape}")
print(df["corpus"].value_counts().to_string())

# ── 3. Top phrases per corpus ─────────────────────────────────────────────────
print()
for corpus_name, group in df.groupby("corpus"):
    print(f"\n{'='*58}")
    print(f" Top 8 Phrases — {corpus_name} Corpus")
    print(f"{'='*58}")
    top = (
        group.sort_values("interestingness", ascending=False)
        .drop_duplicates("phrase").head(8)
        [["phrase", "count_product", "count_total", "interestingness"]]
        .reset_index(drop=True)
    )
    print(top.to_string(index=False))

# ── 4. Sensitivity: exponent sweep ───────────────────────────────────────────
probe_phrases = ["baby shaker", "lost phone", "software update", "data loss"]
probe_df = df.drop_duplicates("phrase").set_index("phrase")
rows_s = []
for phrase in probe_phrases:
    if phrase not in probe_df.index:
        continue
    row = probe_df.loc[phrase]
    for a in [0.50, 0.75, 0.95, 1.00]:
        rows_s.append({
            "phrase": phrase, "alpha": a,
            "score": calculate_interestingness(row["count_product"], row["count_total"], exponent=a),
        })
pivot = pd.DataFrame(rows_s).pivot_table(index="phrase", columns="alpha", values="score")
print("\n=== Exponent Sensitivity (Section 2) ===")
print(pivot.round(4).to_string())

# ── 5. ABLATION STUDY ─────────────────────────────────────────────────────────
noise_words = [
    {"phrase": "the",  "corpus": "iPod", "count_product": 950, "count_total": 500000},
    {"phrase": "and",  "corpus": "iPod", "count_product": 870, "count_total": 480000},
    {"phrase": "a",    "corpus": "iPod", "count_product": 820, "count_total": 460000},
    {"phrase": "is",   "corpus": "iPod", "count_product": 790, "count_total": 440000},
    {"phrase": "this", "corpus": "iPod", "count_product": 750, "count_total": 420000},
]
df_abl = pd.concat([df, pd.DataFrame(noise_words)], ignore_index=True)
df_abl["score_095"] = df_abl.apply(lambda r: calculate_interestingness(r["count_product"], r["count_total"], 0.95), axis=1)
df_abl["score_000"] = df_abl.apply(lambda r: calculate_interestingness(r["count_product"], r["count_total"], 0.0),  axis=1)

ipod_abl = df_abl[df_abl["corpus"] == "iPod"].drop_duplicates("phrase")
top_095 = ipod_abl.nlargest(10, "score_095")[["phrase","count_product","count_total","score_095"]].reset_index(drop=True)
top_000 = ipod_abl.nlargest(10, "score_000")[["phrase","count_product","count_total","score_000"]].reset_index(drop=True)

print("\n=== ABLATION: Top 10 with alpha=0.95 ===")
print(top_095.to_string(index=False))
print("\n=== ABLATION: Top 10 with alpha=0.00 (Pure Volume / Degraded) ===")
print(top_000.to_string(index=False))

# Bar chart
def normalise(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s

union = pd.unique(list(top_095["phrase"]) + list(top_000["phrase"]))
plot_df = ipod_abl[ipod_abl["phrase"].isin(union)].copy()
plot_df["n095"] = normalise(plot_df["score_095"])
plot_df["n000"] = normalise(plot_df["score_000"])
plot_df = plot_df.sort_values("n095", ascending=False)

x = np.arange(len(plot_df))
w = 0.38
fig, ax = plt.subplots(figsize=(14, 6))
b1 = ax.bar(x - w/2, plot_df["n095"], w, label="alpha=0.95 (Section 2)", color="#2196F3", alpha=0.85, edgecolor="white")
b2 = ax.bar(x + w/2, plot_df["n000"], w, label="alpha=0.00 (Pure Volume)", color="#FF5722", alpha=0.85, edgecolor="white")
noise_set = {"the", "and", "a", "is", "this"}
for i, phrase in enumerate(plot_df["phrase"]):
    if phrase in noise_set:
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="red")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["phrase"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Normalised Interestingness Score")
ax.set_title("Ablation Study: alpha=0.95 vs alpha=0.0 (Pure Volume)\nRed shading = noise/filler words that pollute the ablated ranking")
ax.set_ylim(0, 1.15)
noise_patch = mpatches.Patch(color="red", alpha=0.2, label="Noise / filler words")
ax.legend(handles=[b1, b2, noise_patch], fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("results/ablation_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: results/ablation_plot.png")

# ── 6. FAILURE MODE: Data Sparsity ───────────────────────────────────────────
sparse = pd.DataFrame([
    {"phrase": "xyzpod glitch",   "corpus": "iPod", "count_product": 1, "count_total": 1},
    {"phrase": "frobnicate tune", "corpus": "iPod", "count_product": 1, "count_total": 1},
    {"phrase": "wubba lubba",     "corpus": "iPod", "count_product": 1, "count_total": 1},
    {"phrase": "baby shaker",     "corpus": "iPod", "count_product": 82, "count_total": 140},
])
sparse["score_095"]     = sparse.apply(lambda r: calculate_interestingness(r["count_product"], r["count_total"]), axis=1)
sparse["score_smooth"]  = sparse.apply(lambda r: calculate_interestingness_smoothed(r["count_product"], r["count_total"]), axis=1)

print("\n=== FAILURE MODE: Data Sparsity (count_total=1) ===")
print(sparse[["phrase","count_product","count_total","score_095","score_smooth"]].to_string(index=False))
print("\nGhost phrases (count_total=1) score 1.0 without smoothing.")
print("With Laplacian smoothing k=1: ghost scores drop ~0.51, 'baby shaker' correctly tops the list.\n")

# Sparsity chart
fig2, ax2 = plt.subplots(figsize=(8, 4))
xi = np.arange(len(sparse))
w2 = 0.35
ax2.bar(xi - w2/2, sparse["score_095"],    w2, label="Original (no smoothing)", color="#FF5722", alpha=0.85)
ax2.bar(xi + w2/2, sparse["score_smooth"], w2, label="Laplacian k=1",           color="#4CAF50", alpha=0.85)
for i, p in enumerate(sparse["phrase"]):
    if p != "baby shaker":
        ax2.text(i, sparse["score_095"].iloc[i]+0.02, "ghost", ha="center", fontsize=7.5, color="darkred")
ax2.set_xticks(xi)
ax2.set_xticklabels(sparse["phrase"], rotation=20, ha="right", fontsize=9)
ax2.set_ylabel("Interestingness Score")
ax2.set_title("Failure Mode: Laplacian Smoothing (k=1) demotes ghost phrases")
ax2.set_ylim(0, 1.25)
ax2.legend(fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("results/failure_mode_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/failure_mode_plot.png")
print("\nAll done.")
