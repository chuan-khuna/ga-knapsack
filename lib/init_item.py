import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


sample_size = 500
c = 0.1
w = 1.5

weights = np.round(np.random.rand(sample_size)*10 + c, 1)
values = np.round(weights*w + np.random.normal(loc=10, scale=5, size=sample_size), 1)

weights = np.abs(weights)
values = np.abs(values)

fig = plt.figure(figsize=(8, 8), dpi=100)
sns.scatterplot(weights, values)
plt.show()

df = pd.DataFrame({'weight': weights, 'value': values})
df.to_csv("../data/items.csv", index=False)