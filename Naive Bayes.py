#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[52]:


import random
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ## Data Class

# In[53]:


class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else: #Score of 4 or 5
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = [Review(review['reviewText'], review['overall']) for review in reviews]
        
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
        
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)


# ## Load Data

# In[54]:


file_name = r"C:\Users\anush\OneDrive\Desktop\data\sentiments\Books_small.json"

reviews = []
with open(file_name) as f:
    for line in f:
        review_data = json.loads(line)
        reviews.append(review_data)

# Now you can access the review text and overall rating for each review like this:
print(reviews[5]['reviewText'])
print(reviews[5]['overall'])


# ## Prep Data

# In[55]:


training, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(training)

test_container = ReviewContainer(test)
train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))


# ## Bag of words vectorization

# In[56]:


# This book is great !
# This book was so bad

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

print(train_x[0])
print(train_x_vectors[0].toarray())


# ## Classification

# In[57]:


# Create a Multinomial Naive Bayes classifier with Laplace smoothing
alpha_value = 1.0  # You can adjust this value as needed
nb_classifier = MultinomialNB(alpha=alpha_value)

# Fit the classifier to your training data
nb_classifier.fit(train_x_vectors, train_y)

# Make predictions on the test data
test_predictions = nb_classifier.predict(test_x_vectors)


# ## Model Evaluation

# In[58]:


# Evaluate the classifier's performance
print("Accuracy:", accuracy_score(test_y, test_predictions))
print(classification_report(test_y, test_predictions))

# Calculate F1 scores
f1_scores = f1_score(test_y, test_predictions, average=None)
print("F1 Score for Negative Sentiment:", f1_scores[0])
print("F1 Score for Neutral Sentiment:", f1_scores[1])
# Check if there is a third class (positive sentiment) before printing
if len(f1_scores) > 2:
    print("F1 Score for Positive Sentiment:", f1_scores[2])


# ## Prediction using the model

# In[ ]:


# Define the new test set
test_set = ['very fun', "bad book do not buy", 'horrible waste of time']

# Transform the new test set using the same vectorizer
new_test_vectors = vectorizer.transform(test_set)

# Make predictions on the new test set using the trained classifier
new_test_predictions = nb_classifier.predict(new_test_vectors)

# Map predicted labels to sentiment classes based on the classifier's classes
sentiment_mapping = {
    nb_classifier.classes_[0]: Sentiment.NEGATIVE,
    nb_classifier.classes_[1]: Sentiment.NEUTRAL
}

# Map predicted labels to sentiment classes for the new test set
new_test_sentiments = [sentiment_mapping[prediction] for prediction in new_test_predictions]

# Print the predicted sentiments for the new test set
for i, sentiment in enumerate(new_test_sentiments):
    print(f"Test {i+1}: '{test_set[i]}' is predicted as '{sentiment}' sentiment.")


# In[ ]:




