from pathlib import Path

import pandas as pd

from models import build_baseline_cnn

Path("reports").mkdir(exist_ok=True)
model = build_baseline_cnn()
model.summary()

rows = []
for layer in model.layers:
    shape = tuple(layer.output.shape)
    rows.append({
        "layer": layer.name,
        "type": layer.__class__.__name__,
        "output_shape": str(shape),
        "params": layer.count_params(),
    })
table = pd.DataFrame(rows)
table.to_csv("reports/baseline_layers.csv", index=False)
print("\n" + table.to_string(index=False))
print(f"\nTotal parameters: {model.count_params():,}")
print(f"Trainable: {sum(int(w.numpy().size) for w in model.trainable_weights):,}")
print(f"Non-trainable: {sum(int(w.numpy().size) for w in model.non_trainable_weights):,}")