import pandas as pd

df = pd.read_csv('sample_full.csv')
df['label'] = df.iloc[:, 1:].idxmax(axis=1)
# rename Learning_outcome column to Text
df.rename(columns={'Learning_outcome': 'Text'}, inplace=True)

# drop all columns except Text and label
df = df[['Text', 'label']]
df.to_csv('sample_full_cleaned.csv', index=False)
