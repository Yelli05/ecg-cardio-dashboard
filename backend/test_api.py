import pandas as pd
import requests

URL = "http://127.0.0.1:8000"
print("health:", requests.get(f"{URL}/health").json())

df = pd.read_csv("data/mitbih_test.csv", header=None)
sample = df.groupby(187).head(2)          # first 2 beats of each class = 10 beats

for _, row in sample.iterrows():
    r = requests.post(f"{URL}/predict", json={"values": row.iloc[:187].tolist()})
    out = r.json()
    print(f"true={int(row.iloc[187])}  predicted={out['predicted_class']}  "
          f"{out['label']:<30} confidence={out['confidence']:.3f}")

# A bad request: only 10 values instead of 187
bad = requests.post(f"{URL}/predict", json={"values": [0.1] * 10})
print("bad input status:", bad.status_code)