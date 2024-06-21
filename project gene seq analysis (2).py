#!/usr/bin/env python
# coding: utf-8

# # E.COLI PROMOTER GENE SEQUENCE  ANALYSIS

# In[1]:


import pandas as pd

# Define the path to the data file
data_file_path = 'promoters[1].data'

# Load the data into a DataFrame
data = []
with open(data_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        label = 1 if parts[0] == '+' else 0  # Convert class label to binary
        instance_name = parts[1]
        sequence = parts[2].replace("\t", "")  # Remove any tab characters in the sequence
        data.append([label, instance_name, sequence])

df = pd.DataFrame(data, columns=['label', 'instance_name', 'sequence'])

# Display the first few rows
print(df.head())


# In[2]:


# Display basic information about the DataFrame
print(df.info())

# Display basic statistics
print(df.describe())

# Check the class distribution
print(df['label'].value_counts())


# # EXPLORATORY DATA ANALYSIS

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the distribution of each nucleotide across all sequences
nucleotide_counts = df['sequence'].apply(lambda seq: pd.Series(list(seq))).stack().value_counts(normalize=True)

# Plot the nucleotide distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=nucleotide_counts.index, y=nucleotide_counts.values)
plt.title('Nucleotide Distribution')
plt.xlabel('Nucleotide')
plt.ylabel('Frequency')
plt.show()


# In[4]:


# Create a DataFrame where each column is a nucleotide position
sequence_df = df['sequence'].apply(lambda x: pd.Series(list(x)))

# Add the label column
sequence_df['label'] = df['label']

# Calculate the frequency of each nucleotide at each position for promoters
promoter_sequences = sequence_df[sequence_df['label'] == 1].drop(columns=['label'])
promoter_nucleotide_freq = promoter_sequences.apply(lambda col: col.value_counts(normalize=True)).transpose()

# Calculate the frequency of each nucleotide at each position for non-promoters
non_promoter_sequences = sequence_df[sequence_df['label'] == 0].drop(columns=['label'])
non_promoter_nucleotide_freq = non_promoter_sequences.apply(lambda col: col.value_counts(normalize=True)).transpose()

# Plot the nucleotide frequency at each position for promoters
plt.figure(figsize=(20, 6))
sns.heatmap(promoter_nucleotide_freq, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title('Nucleotide Frequency at Each Position (Promoters)')
plt.xlabel('Nucleotide')
plt.ylabel('Position')
plt.show()

# Plot the nucleotide frequency at each position for non-promoters
plt.figure(figsize=(20, 6))
sns.heatmap(non_promoter_nucleotide_freq, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title('Nucleotide Frequency at Each Position (Non-Promoters)')
plt.xlabel('Nucleotide')
plt.ylabel('Position')
plt.show()


# In[5]:


def plot_class_distribution():
    plt.figure(figsize=(6, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Class (0 = Non-Promoter, 1 = Promoter)')
    plt.ylabel('Count')
    plt.show()


# In[6]:


plot_class_distribution()


# In[7]:


def plot_sequence_length_distribution():
    sequence_lengths = df['sequence'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(sequence_lengths, kde=True, bins=10, color='purple')
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.show()


# In[8]:


plot_sequence_length_distribution()


# In[9]:


# Function to calculate GC content
def gc_content(sequence):
    return (sequence.count('g') + sequence.count('c')) / len(sequence)

# Calculate GC content for each sequence
df['gc_content'] = df['sequence'].apply(gc_content)

# Plot GC content distribution for promoters and non-promoters
plt.figure(figsize=(10, 6))
sns.histplot(df[df['label'] == 1]['gc_content'], color='blue', kde=True, label='Promoters')
sns.histplot(df[df['label'] == 0]['gc_content'], color='red', kde=True, label='Non-Promoters')
plt.title('GC Content Distribution')
plt.xlabel('GC Content')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[10]:


from scipy.stats import entropy

# Calculate the information content (entropy) at each position
promoter_entropy = promoter_nucleotide_freq.apply(entropy, axis=1)
non_promoter_entropy = non_promoter_nucleotide_freq.apply(entropy, axis=1)

# Plot the information content
plt.figure(figsize=(20, 6))
plt.plot(promoter_entropy, label='Promoters', color='blue')
plt.plot(non_promoter_entropy, label='Non-Promoters', color='red')
plt.title('Information Content at Each Position')
plt.xlabel('Position')
plt.ylabel('Entropy')
plt.legend()
plt.show()


# # CLASSIFICATION ALGORITHMS

# In[11]:


import numpy as np 

names = ['Class', 'id', 'Sequence']
data = pd.read_csv("promoters[1].data",names = names)


# In[12]:


data


# In[13]:


print(data.iloc[0])


# In[14]:


classes = data.loc[:, 'Class']
print(classes[:5])


# In[15]:


sequences = list(data.loc[:, 'Sequence'])
dataset = {}

# loop through sequences and split into individual nucleotides
for i, seq in enumerate(sequences):
    
    # split into nucleotides, remove tab characters
    nucleotides = list(seq)
    nucleotides = [x for x in nucleotides if x != '\t']
    
    # append class assignment
    nucleotides.append(classes[i])
    
    # add to dataset
    dataset[i] = nucleotides
    
print(dataset[0])


# In[16]:


# turn dataset into pandas DataFrame
dframe = pd.DataFrame(dataset)
print(dframe)


# In[17]:


df = dframe.transpose()
print(df.iloc[:5])


# In[18]:


# for clarity, lets rename the last dataframe column to class
df.rename(columns = {57: 'Class'}, inplace = True) 
print(df.iloc[:5])


# In[19]:


df.describe()


# In[20]:


# desribe does not tell us enough information since the attributes are text. Lets record value counts for each sequence
series = []
for name in df.columns:
    series.append(df[name].value_counts())
    
info = pd.DataFrame(series)
details = info.transpose()
print(details)


# In[21]:


# Unfortunately, we can't run machine learning algorithms on the data in 'String' formats. As a result, we need to switch
# it to numerical data. This can easily be accomplished using the pd.get_dummies() function
numerical_df = pd.get_dummies(df)
numerical_df.iloc[:5]


# In[22]:


# We don't need both class columns.  Lets drop one then rename the other to simply 'Class'.
df = numerical_df.drop(columns=['Class_-'])

df.rename(columns = {'Class_+': 'Class'}, inplace = True)
print(df.iloc[:5])


# In[23]:


# Use the model_selection module to separate training and testing datasets
from sklearn import model_selection

# Create X and Y datasets for training
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

# define seed for reproducibility
seed = 1

# split data into training and testing datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=seed)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# define scoring method
scoring = 'accuracy'

# Define models to train
names = ["Nearest Neighbors", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "SVM Linear", "SVM RBF", "SVM Sigmoid"]

classifiers = [
    KNeighborsClassifier(n_neighbors = 3),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(kernel = 'linear'), 
    SVC(kernel = 'rbf'),
    SVC(kernel = 'sigmoid')
]

models = zip(names, classifiers)

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('Test-- ',name,': ',accuracy_score(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))


# # LINEAR REGRESSION TO PREDICT GC CONTENT

# In[25]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define the path to the data file
data_file_path = 'promoters[1].data'

# Load the data into a DataFrame
data = []
with open(data_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        label = 1 if parts[0] == '+' else 0  # Convert class label to binary
        instance_name = parts[1]
        sequence = parts[2].replace("\t", "")  # Remove any tab characters in the sequence
        data.append([label, instance_name, sequence])

df = pd.DataFrame(data, columns=['label', 'instance_name', 'sequence'])

# Function to calculate GC content
def gc_content(sequence):
    return (sequence.count('g') + sequence.count('c')) / len(sequence)

# Calculate GC content for each sequence
df['gc_content'] = df['sequence'].apply(gc_content)

# Create a DataFrame where each column is a nucleotide position
sequence_df = df['sequence'].apply(lambda x: pd.Series(list(x)))

# One-hot encode the nucleotide positions
encoder = OneHotEncoder(sparse=False)
encoded_sequences = encoder.fit_transform(sequence_df)

# Combine the encoded sequences with the GC content
X = encoded_sequences
y = df['gc_content']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the GC content on the test set
y_pred = model.predict(X_test)
print(y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Optional: Compare predicted vs actual values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())


# In[ ]:




