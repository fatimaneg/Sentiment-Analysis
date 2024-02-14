#SENTIMENT ANALYSIS OF AMAZON FOOD REVIEWS
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

#Set a stylesheet for plots 
plt.style.use('ggplot')
import nltk

#Read in data
df = pd.read_csv(('/Users/fatimaisaeva/Downloads/Reviews.csv'))
print(df.head())

#Reducing analysis to the first 500 reviews 
df = df.head(500)
print(df.shape)

#PLOT the count of reviews by stars
# Calculate the occurrence of each unique value in the column 'Score' with .value_counts()
ax = sns.countplot(data=df, x='Score')
ax.set_title('Count of Reviews by Stars')
ax.set_xlabel('Review Stars')
ax.set_ylabel('Count')
#plt.show() #most reviews are positive 

#NLTK processing 
#example of a negtive review 
review_negative = df['Text'][50]
print(review_negative)
tokens = nltk.word_tokenize(review_negative)

print(tokens[0:10])

#Part of speech tagging on negative review
tagged = nltk.pos_tag(tokens)

#Chunk the list of tokens according to pos
chunked = nltk.chunk.ne_chunk(tagged)
print(chunked)

#VADER Sentiment Scoring (positive/ negative/ neutral words) on a scale from 0 to 1
#VADER doesn't account for relationships between words

#Small practice
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm # to visually track the progress of loops and iterations

sia = SentimentIntensityAnalyzer()

example_positive = sia.polarity_scores('I am so happy!')
print(example_positive) # on a scale from 0 to 1 neg=0.0, pos = 0.682

example_negative = sia.polarity_scores('This is the worst ricecooker ever')
print(example_negative) # neg = 0.451, pos = 0.0

#Run SIA on 1 Amazon Review
sia_review = sia.polarity_scores(review_negative)
print(sia_review) #neg-0.22, pos-0.0

#Run the ploarity score on the entire dataset 
result = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
    text = row['Text']
    myid = row['Id']
    result[myid] = sia.polarity_scores(text)

#Turn to pandas frame for visual convenience 
vaders = pd.DataFrame(result).T
print(vaders.head())

#Rename the column name for further merge
vaders = vaders.reset_index().rename(columns={'index': 'Id'})

#Merge on the original dataframe 'df' - left merge
vaders = vaders.merge(df, how='left')
print(vaders.head())

#Plot results - correlation of score and compound of sia
plt.style.use('ggplot')
ax = sns.barplot(data = vaders, x ='Score', y = 'compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()


# Comparing three plots side by side (positive, negative, neutral scores)
fig, axs = plt.subplots(1, 3, figsize = (15, 5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
#plt.show()
# Resulting plots validate the VADER reliability


# More elaborate analysis using ROBERTA PRETRAINED MODEL (WORD CONTEXT)
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax #converts scores into probabilities

#Use pre-trained model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment" #model pre-trained on Twitter data 
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL) 

#Run Roberta Model on 1 Amazon review
print(review_negative)
encoded_text = tokenizer(review_negative, return_tensors = 'pt')
output = model (**encoded_text)
scores = output[0][0].detach().numpy() #detach a tensor from the computation graph and convert it to a NumPy array 
scores = softmax(scores) # normalize scores
print(scores)

scores_dict = {
    'roberta_neg' : scores[0], 
    'roberta_neu' : scores [1], 
    'roberta_pos': scores[2]
} #keys - sentiment categories, values - corressponding probabilities in the array

print(scores_dict) #negative score much higher than in VADERS -> more powerful than VADERS

#Run ROBERTA MODEL on the enire dataset
#Define a function to yield scores for Roberta model
def polarity_scores_roberta(example): 
    encoded_text = tokenizer(review_negative, return_tensors = 'pt')
    output = model (**encoded_text)
    scores = output[0][0].detach().numpy() #detach a tensor from the computation graph and convert it to a NumPy array 
    scores = softmax(scores) # normalize scores
    scores_dict = {
    'roberta_neg' : scores[0], 
    'roberta_neu' : scores [1], 
    'roberta_pos': scores[2]
    }

    return scores_dict

#Iterating over each score in the column with Vader + with Roberta model
#Some lines of text failed to run through Roberta model, skip over them
aggreg_result = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
    try:
        text = row['Text'] 
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both_models = {**vader_result_rename, **roberta_result}
        aggreg_result[myid] = both_models
    except RuntimeError: 
        print(f"Broke for id {myid}")

#Merging the result of the dictionary to the existing dataframe with reviews 
results_df = pd.DataFrame(aggreg_result).T
results_df = results_df.reset_index().rename(columns= {'index': 'Id'})
results_df = results_df.merge(df, how = 'left')

#Print out resulting data frame head 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(results_df.head())

#Compare scores between plots-pairplot 
sns.pairplot(data = results_df, 
             vars = ['vader_neg', 'vader_neu', 'vader_pos', 
                    'roberta_neg', 'roberta_neu', 'roberta_pos'],
                    hue = 'Score', palette= 'tab10')
plt.show()