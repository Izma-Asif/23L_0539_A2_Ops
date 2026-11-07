from pathlib import Path
import matplotlib
matplotlib.use("Agg")             
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

SEED = 42
N_PER_CLASS = 8
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("reports/audit.csv")
df = df[~df["corrupt"]].copy()
df["class"] = df["class"].map({"ai": "AI-generated", "real": "Real"})
palette = {"AI-generated": "#d95f02", "Real": "#1b9e77"}

fig, axes = plt.subplots(2, N_PER_CLASS, figsize=(2.2 * N_PER_CLASS, 5.4))
for r, cls in enumerate(["AI-generated", "Real"]):
    sample = df[df["class"] == cls].sample(min(N_PER_CLASS, (df["class"] == cls).sum()), random_state=SEED)
    for c in range(N_PER_CLASS):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(sample):
            row = sample.iloc[c]
            with Image.open(row["path"]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{int(row['width'])}x{int(row['height'])}", fontsize=8)
    axes[r, 0].text(-0.08, 0.5, cls, transform=axes[r, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=12, fontweight="bold")
fig.suptitle("Random sample per class (title = width x height in pixels)")
fig.tight_layout()
fig.savefig(OUT / "fig1_sample_grid.png", dpi=150)
plt.close(fig)

specs = [
    ("width", "Width (pixels, log scale)", "fig2_width_hist.png"),
    ("height", "Height (pixels, log scale)", "fig3_height_hist.png"),
    ("file_size", "File size (bytes, log scale)", "fig4_filesize_hist.png"),
]
for col, xlabel, fname in specs:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x=col, hue="class", log_scale=True, bins=40, stat="density",
                 common_norm=False, element="step", palette=palette, ax=ax)
    for cls, g in df.groupby("class"):                  # dashed line = class median
        ax.axvline(g[col].median(), color=palette[cls], linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density (each class sums to 1)")
    ax.set_title(f"{col.replace('_', ' ').title()} per class (dashed = median)")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)

print("Saved:", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")