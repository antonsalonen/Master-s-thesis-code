import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = "/Users/antonsalonen/Desktop/DATASET0.xlsx"

VARIABLES = [
    "Open",
    "Close",
    "VSTOXX",
    "EUR/USD",
    "ECB main refinancing rate",
    "Consumer confidence",
    "Unemployment rate"
]

df = pd.read_excel(FILE_PATH)
df = df.set_index("Date")

data = df[VARIABLES]

correlation_matrix = data.corr(method="pearson")

plt.figure(figsize=(9,6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True
)
plt.tight_layout()
plt.show()