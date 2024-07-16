from datasets import load_dataset
import pandas as pd
print(1)
ds = load_dataset("nkp37/OpenVid-1M")
print(2)
df=pd.read_csv(ds)
print(3)
df.to_csv('datacard.csv')