# Upgrade gensim as needed (uncomment if required)
#!pip install --upgrade gensim

# Import dependencies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score


# Load full dataset (we use nrows=5000 for speed)
data_url = "https://raw.githubusercontent.com/TAUforPython/BioMedAI/main/test_datasets/test_data_ECG.csv"
raw_table_data = pd.read_csv(data_url, nrows=5000)
print("Data loaded. Shape:", raw_table_data.shape)
print(raw_table_data.head(3))


# Remove outliers: only keep rows where all selected numeric columns are < 2000.
columns_to_filter = ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 
                     'p_axis', 'qrs_axis', 't_axis']
full_df_filtered = raw_table_data[(raw_table_data[columns_to_filter] < 2000).all(axis=1)]

# Further filter: ensure that p_onset < p_end and qrs_onset < qrs_end
full_df_filtered = full_df_filtered[
    (full_df_filtered['p_onset'] < full_df_filtered['p_end']) & 
    (full_df_filtered['qrs_onset'] < full_df_filtered['qrs_end'])
]

# Merge all text reports into one field.
reports = [f'report_{x}' for x in range(18)]
full_df_filtered['report_0'] = full_df_filtered[reports].astype(str).agg(' '.join, axis=1)
full_df_filtered['report_0'] = full_df_filtered['report_0'].str.replace(r'\bnan\b', '', regex=True)\
                                                       .str.replace(r'\s+', ' ', regex=True)\
                                                       .str.strip()
full_df_filtered.rename(columns={'report_0': 'report'}, inplace=True)
# Drop the other report columns (report_1 to report_17)
reports_to_drop = [f'report_{x}' for x in range(1, 18)]
full_df_filtered = full_df_filtered.drop(reports_to_drop, axis=1)

# Fix column names – remove trailing spaces from 'eeg_time ' and 'eeg_date '
full_df_filtered = full_df_filtered.rename(columns={'eeg_time ': 'eeg_time', 'eeg_date ': 'eeg_date'})

# Drop unnecessary columns: bandwidth and filtering
for col in ['bandwidth', 'filtering']:
    if col in full_df_filtered.columns:
        full_df_filtered = full_df_filtered.drop(columns=[col])

# Rearrange columns so that Healthy_Status is the last column.
cols_order = [col for col in full_df_filtered.columns if col != 'Healthy_Status'] + ['Healthy_Status']
full_df_filtered = full_df_filtered[cols_order]


# Tokenize report text (split by whitespace)
words = [text.split() for text in full_df_filtered['report']]

# Train Word2Vec model on the tokens (using default parameters; adjust if needed)
w2v_model = Word2Vec(words, min_count=1)

# Define function that returns the average embedding for a sentence.
def get_sentence_embedding(sentence):
    tokens = sentence.split()
    # For each token that is in the vocabulary get its vector
    word_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)
    
# Apply the function to each report, then take its scalar mean.
# (That is, for each embedding vector, we calculate mean over its components.)
full_df_filtered['report'] = full_df_filtered['report'].apply(lambda x: get_sentence_embedding(x).mean())


table_data = full_df_filtered[['report','rr_interval','p_end','qrs_onset','qrs_end','t_end',
                               'p_axis','qrs_axis','t_axis','Healthy_Status']].copy()
print("Prepared table_data shape:", table_data.shape)
print(table_data.head(3))

X = table_data.drop(columns=['Healthy_Status'])
y = table_data['Healthy_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Train set shape:", X_train.shape, "Test set shape:", X_test.shape)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ada', AdaBoostClassifier(algorithm='SAMME', random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("AdaBoost Classifier Performance:")
print("Accuracy: {:.4f}".format(acc))
print("F1 Score: {:.4f}".format(f1))
print("AUC: {:.4f}".format(auc))
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:\n", report)

# For example, a boxplot of a few features to check distributions post-cleaning
plt.figure(figsize=(12, 6))
sns.boxplot(data=table_data.drop(columns=['Healthy_Status']))
plt.title("Boxplot of ECG Features (and Report Embedding)")
plt.xticks(rotation=45)
plt.show()

# Plot the distribution of the target Healthy_Status
plt.figure(figsize=(6,4))
sns.countplot(x="Healthy_Status", data=table_data)
plt.title("Distribution of Healthy_Status")
plt.show()
