import pandas as pd

df = pd.read_csv("data/mitbih_test.csv", header=None)
samples = df.groupby(187).sample(4, random_state=1)      # 4 beats per class = 20 rows
samples = samples.sample(frac=1, random_state=1)         # shuffle the order
samples.columns = [f"t{i}" for i in range(187)] + ["label"]
samples["label"] = samples["label"].astype(int)
samples.to_csv("frontend/sample_heartbeats.csv", index=False)
print(samples.shape, samples["label"].value_counts().sort_index().to_dict())