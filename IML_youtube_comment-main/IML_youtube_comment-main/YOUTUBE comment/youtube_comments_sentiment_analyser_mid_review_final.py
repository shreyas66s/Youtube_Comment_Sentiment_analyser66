
!pip install emoji

import os
import pandas as pd
from googleapiclient.discovery import build
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.metrics import accuracy_score, classification_report
import emoji
import re
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

api_key = "AIzaSyBtpjwDRl5oP1RJgANjgMOAZaXYAwIM_Q4"
youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_comments(video_id):
    comments = []
    likes_counts = []
    replies_counts = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=2500  # Adjust as per your requirements
    )

    while request:
        response = request.execute()
        for item in response["items"]:
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            likes_count = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
            replies_count = item["snippet"]["totalReplyCount"]  # Includes replies count

            # Append data to respective lists
            comments.append(comment_text)
            likes_counts.append(likes_count)
            replies_counts.append(replies_count)

        request = youtube.commentThreads().list_next(request, response)

    return comments, likes_counts, replies_counts

video_id = input("Enter the YouTube video ID: ")
comments_texts,likes_counts,replies_counts = get_video_comments(video_id)

df = pd.DataFrame({
    'Comment': comments_texts,
    'LikesCount': likes_counts,
    'RepliesCount': replies_counts
})

df

len=df.shape[0]
print(len)

def remove_emojis(text):
    text_without_emojis = emoji.demojize(text)
    return re.sub(r':[a-z_]+:', '', text_without_emojis)

df['Comment'] = df['Comment'].apply(remove_emojis)

# Remove non-printable characters
df = df.applymap(lambda x: ''.join([c for c in str(x) if c.isprintable()]))

output_file = 'output.xlsx'
df.to_excel(output_file, index=False)

df['Comment'] = df['Comment'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower().strip()))

df.head()

df

nltk.download('punkt')
nltk.download('stopwords')

lancaster = LancasterStemmer()
stop_words = set(stopwords.words('english'))

df['Comment'] = df['Comment'].apply(lambda x: word_tokenize(str(x)))

df['Comment'] = df['Comment'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

df['Comment'] = df['Comment'].apply(lambda tokens: [lancaster.stem(token) for token in tokens])

df['Comment'] = df['Comment'].apply(lambda tokens: ' '.join(tokens))

df.head(10)

def get_sentiment(comment):
    analysis = TextBlob(comment)
    return analysis.sentiment.polarity

df['Label'] = df['Comment'].apply(get_sentiment)
df['Label'] = df['Label'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

sentiment_counts = df['Label'].value_counts()
print(sentiment_counts)

df

# Count the number of rows with 'neutral' class
neutral_count = (df['Label'] == 'neutral').sum()

# Determine the number of rows to delete
rows_to_delete = min(200, neutral_count)

# Get the indices of rows with 'neutral' class
indices_to_delete = df[df['Label'] == 'neutral'].index[:rows_to_delete]

# Drop the rows with the specified indices
df = df.drop(indices_to_delete)

sentiment_counts = df['Label'].value_counts()
print(sentiment_counts)

"""RANDOM FOREST MODEL"""

X_train, X_test, y_train, y_test = train_test_split(df[['Comment','LikesCount','RepliesCount']], df['Label'], test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train['Comment'])
X_test_tfidf = vectorizer.transform(X_test['Comment'])

# rf_classifier = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=30, min_samples_split=3,min_samples_leaf=1)
rf_classifier = RandomForestClassifier(n_estimators=300, criterion="gini",max_depth=50,min_samples_split=7,min_samples_leaf=2)
#rf_classifier = RandomForestClassifier(n_estimators=600, criterion="gini", class_weight={"negative":3,"neutral":1,"positive":1},max_depth=15,min_samples_split=2,min_samples_leaf=2)
rf_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the training data
y_train_pred = rf_classifier.predict(X_train_tfidf)

# Calculate training accuracy
accuracy_rf = accuracy_score(y_train, y_train_pred)
print(f'Random Forest Classifier Training Accuracy: {accuracy_rf:.2f}')

y_pred_rf = rf_classifier.predict(X_test_tfidf)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Classifier Accuracy: {accuracy_rf:.2f}')

print('\nClassification Report (Random Forest Classifier):')
print(classification_report(y_test, y_pred_rf))

df

"""AUC CALCULATION"""

lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

y_scores = rf_classifier.predict_proba(X_test_tfidf)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in enumerate(colors[:3]):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(lb.classes_[i], roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
         label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

for i in range(3):
    print(f'AUC for class {lb.classes_[i]}: {roc_auc[i]:.2f}')
print(f'Micro-average AUC: {roc_auc["micro"]:.2f}')

cm=confusion_matrix(y_test,y_pred_rf)

print("Confusion Matrix:")
print(cm)

import os
import pandas as pd
from googleapiclient.discovery import build
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.metrics import accuracy_score, classification_report
import emoji
import re
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

api_key = "AIzaSyAmwO87l2eymyoLCCEze9Os6PSLA6pmftg"

youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=5000
    )
    while request:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    return comments

video_id = input("Enter the YouTube video ID: ")

comments = get_video_comments(video_id)

df = pd.DataFrame({'Comment': comments})

def remove_emojis(text):
    text_without_emojis = emoji.demojize(text)
    return re.sub(r':[a-z_]+:', '', text_without_emojis)

df['Comment'] = df['Comment'].apply(remove_emojis)

output_file = 'output.xlsx'
df.to_excel(output_file, index=False)

df['Comment'] = df['Comment'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower().strip()))

nltk.download('punkt')
nltk.download('stopwords')

lancaster = LancasterStemmer()
stop_words = set(stopwords.words('english'))

df['Comment'] = df['Comment'].apply(lambda x: word_tokenize(str(x)))

df['Comment'] = df['Comment'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

df['Comment'] = df['Comment'].apply(lambda tokens: [lancaster.stem(token) for token in tokens])

df['Comment'] = df['Comment'].apply(lambda tokens: ' '.join(tokens))

df.head(10)

def get_sentiment(comment):
    analysis = TextBlob(comment)
    return analysis.sentiment.polarity

df['Sentiment'] = df['Comment'].apply(get_sentiment)

df['Label'] = df['Sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

sentiment_counts = df['Label'].value_counts()
print(sentiment_counts)

X_train, X_test, y_train, y_test = train_test_split(df['Comment'], df['Label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear'),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

for clf_name, clf in classifiers.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n{clf_name} Classifier Accuracy: {accuracy:.2f}')
    print(f'\nClassification Report ({clf_name} Classifier):')
    print(classification_report(y_test, y_pred))

"""****GridSearchCV****"""

from sklearn.model_selection import GridSearchCV
import numpy as np
param_grid = {
    'n_estimators': [100,200,300,400,500,600,700,800],
    # 'criterion':['gini'],
    # 'max_depth': [10,15,20],  # Increase max_depth
    # # 'min_samples_split': [2,5],  # Decrease min_samples_split
    # # 'min_samples_leaf': [1,2],  # Decrease min_samples_leaf
    # 'max_features': [0.5,1,2,3],  # Increase max_features
    # 'bootstrap': [True, False],
}

unique_classes = np.unique(y_train)
print(unique_classes)

class_weight = {'negative': 2, 'neutral': 1, 'positive': 1}
if set(unique_classes) == set(class_weight.keys()):
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    best_rf_classifier = grid_search.fit(X_train_tfidf, y_train)
    y_pred_best = best_rf_classifier.predict(X_test_tfidf)
    print('\nClassification Report (Best Random Forest Classifier):')
    print(classification_report(y_test, y_pred_best))
else:
    print("Class labels in y_train do not match with class_weight keys.")

grid_search.best_params_

from sklearn.model_selection import GridSearchCV
import numpy as np
param_grid = {
    'n_estimators': [600],
    'criterion':['gini'],
    'max_depth': [140],  # Increase max_depth
    'min_samples_split': [6],  # Decrease min_samples_split
    'min_samples_leaf': [2],  # Decrease min_samples_leaf
    'max_depth':[60],
    'max_leaf_nodes':[110],
    # 'max_features': [0.5,1,2,3],  # Increase max_features
    'bootstrap': [False],
}

unique_classes = np.unique(y_train)
print(unique_classes)

class_weight = {'negative':6, 'neutral':5, 'positive':5}
# if set(unique_classes) == set:
grid_search = GridSearchCV(RandomForestClassifier(random_state=42,class_weight=class_weight), param_grid, cv=5)
best_rf_classifier = grid_search.fit(X_train_tfidf, y_train)
y_pred_best = best_rf_classifier.predict(X_test_tfidf)
print('\nClassification Report (Best Random Forest Classifier):')
print(classification_report(y_test, y_pred_best))
# else:
#     print("Class labels in y_train do not match with class_weight keys.")

# Make predictions on the training data
y_train_pred = best_rf_classifier.predict(X_train_tfidf)

# Calculate training accuracy
accuracy_rf = accuracy_score(y_train, y_train_pred)
print(f'Random Forest Classifier Training Accuracy: {accuracy_rf:.2f}')

accuracy = accuracy_score(y_test, y_pred_best)
print(f'\nClassifier Accuracy: {accuracy:.2f}')

grid_search.best_params_

"""**RandomSearchCV**"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

class_weights = {'negative': 20, 'neutral': 15, 'positive': 10}

param_dist = {
    'n_estimators': [600, 700, 800],
    'max_depth': [None, 10, 20],
    'class_weight': [class_weights, 'balanced'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train_tfidf, y_train)

best_rf_classifier = random_search.best_estimator_
best_rf_classifier.fit(X_train_tfidf, y_train)

y_pred_best = best_rf_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred_best)
print(f'\nClassifier Accuracy: {accuracy:.2f}')

print('Classification Report (Best Estimator):')
print(classification_report(y_test, y_pred_best))



