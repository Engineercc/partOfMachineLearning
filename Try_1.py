import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
# Train data set open
df=pd.read_csv("training_set.txt", sep='\t', names=['liked', 'txt'])
# Test data open
# df1 = pd.read_csv('tweet_s.csv')
# print(df1.head())
# data Cleaning with StopWords
stop_set = set(stopwords.words('turkish'))
vectorizer = TfidfVectorizer(use_idf=True,  lowercase=True, strip_accents='ascii', stop_words=stop_set)

y= df.liked
X=vectorizer.fit_transform(df.txt)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)
print((roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])))

with open('test_set.txt', 'r') as f:
    results = [line.strip().lower() for line in f if line]

tweet_analysis_array = pd.np.array(results)
tweet_analysis_vektor = vectorizer.transform(tweet_analysis_array)
count_neg= 0
count_pos= 0
start=0
c_pos=[]
c_neg=[]
len_arr=len(tweet_analysis_array)
arr=clf.predict(tweet_analysis_vektor)
# print(arr)
# print(results)

while start < len_arr:

    if arr[start] == 1:
        c_pos.append(1)
        count_pos += 1
        start += 1


    else:
        count_neg += 1
        c_neg.append(0)
        start += 1




print('pozitif tweet sayısı = '+str(count_pos))
print('negatif tweet sayısı = '+str(count_neg))

# Get current size
fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]
# print("Current size:", fig_size)

# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

# CHART_PART

slices=[count_pos, count_neg]
pie_lable = ['Positive Tweets','Negative Tweets']
colours=['lightcoral', 'lightskyblue']
plt.pie(slices,
        labels=pie_lable,
        colors=colours,
        shadow=True,
        explode=(0.1, 0),
        autopct='%1.1f%%',
        )
plt.title('TWEET ANALYSIS RESULTS')
plt.legend()
plt.show()
