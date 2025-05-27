import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# 1) Load the CSV you generated
df = pd.read_csv("scores.csv")

# 2) Define the grids you want to try
sim_threshs = np.linspace(0.60, 0.80, 5)     # e.g. [0.60, 0.65, 0.70, 0.75, 0.80]
clf_threshs = np.linspace(0.45, 0.65, 5)     # e.g. [0.45, 0.50, 0.55, 0.60, 0.65]

results = []
for sim_t in sim_threshs:
    # first filter out any candidate below sim threshold
    sub = df[df["sim"] >= sim_t]
    for clf_t in clf_threshs:
        preds = (sub["prob"] >= clf_t).astype(int)
        # compare against true labels
        p, r, f1, _ = precision_recall_fscore_support(
            sub["label"], preds, average="binary", zero_division=0
        )
        results.append({
            "sim_t": sim_t,
            "clf_t": clf_t,
            "precision": p,
            "recall": r,
            "f1": f1,
            "num_pairs": len(sub)
        })

res_df = pd.DataFrame(results)
res_df.to_csv("threshold_sweep_results.csv", index=False)
print(res_df)
