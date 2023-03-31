#!/usr/bin/env python
# coding: utf-8

# # SANDEEP KOLAPINENI (C0827402)

# In[23]:


import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import requests
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


# ## Data Collection 

# In[3]:


# USING VIDEO ID and API KEY fetch video details for analysis, youtube give below details, we created them as a Dictionary and returning 



def fetch_video_details(video_id):

    # Set up the YouTube API URL with the video ID and API key 
    url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={API_KEY}'

    try:
        # Make the API request and convert the response to JSON
        response = requests.get(url).json()

        # If there are no items in the response or the first item is missing required fields, return None
        if not response.get('items') or not all(
                key in response['items'][0] for key in ('snippet', 'statistics', 'contentDetails')):
            return None

        # Extract the necessary details from the response
        video = response['items'][0]
        snippet = video['snippet']
        statistics = video['statistics']
        content_details = video['contentDetails']

        # Extract the comments, if any
        comment_url = f'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key=key={API_KEY}'
        comment_response = requests.get(comment_url).json()
        comments = [comment['snippet']['topLevelComment']['snippet']['textDisplay'] for comment in
                    comment_response.get('items', [])]

        # Build a dictionary containing all the video details
         # Build a dictionary containing all the video details
        details = {
            'video_id': video_id,
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'dislike_count': int(statistics.get('dislikeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'favorite_count': int(statistics.get('favoriteCount', 0)),
            'duration': content_details.get('duration', ''),
            'comments': comments
        }
      
        return details

    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the API request
        print(f"An exception occurred for video {video_id}: {e}")
        return None


# In[ ]:


#READ video ids from given file 
df = pd.read_csv('vdoLinks.csv')
#Youtube only provide 10000 hits , 5000 for video 5000 hits for comments 
#  Select the  5000 rows each time  in the video_id column
video_ids = df['video_id'].iloc[0:25600]

# Apply fetch_video_details function to each id and drop null values
df_details = video_ids.apply(fetch_video_details).dropna()

# Create a DataFrame from the fetched details
df_details = pd.DataFrame(df_details.tolist(), columns=['video_id', 'title', 'description',
                                                        'view_count',
                                                        'like_count', 'dislike_count', 'comment_count', 'duration',
                                                        'comments'])

# Save the details to file
df_details.to_csv('fetched_video_details.csv', index=False,mode='a',header=False)


# In[3]:


# load the data for analysis 
df_details = pd.read_csv('fetched_video_details.csv')
df_details.head()


# In[6]:


df_details = df_details.drop_duplicates(subset='video_id', keep='first')


# ## Top 10 videos

# In[8]:


top10 = df_details.sort_values(by='view_count', ascending=False).head(10)
top10[['video_id', 'title', 'view_count']]


# In[14]:


titles = top10['title']

views = top10['view_count']/10000000

# TOP 10 vides
plt.bar(titles, views,color ='maroon',
        width = 0.4)
plt.xticks(rotation=90)
plt.xlabel('Video Title')
plt.ylabel('View Count(Millions)')
plt.title('Top 10 viewed Videos')
plt.show()


# ##  last 10 viewed videos

# In[10]:


last10 = df_details.loc[df_details["view_count"] >0 ].sort_values(by='view_count', ascending=True).head(10)
last10[['video_id', 'title', 'view_count']]


# In[15]:


titles = last10['title']
views = last10['view_count']


plt.bar(titles, views,color ='red',
        width = 0.4)
plt.xticks(rotation=90)
plt.xlabel('Video Title')
plt.ylabel('View Count')
plt.title('Least 10 viewed Videos ')
plt.show()


# ## MOST LIKED VIDEO

# In[16]:


most_liked = df_details.sort_values(by='like_count', ascending=False).iloc[0]
most_liked[['video_id',  'title','like_count']]
titles = most_liked['title']
likes = most_liked['like_count']
print("MOST LIKED VIDEO ",titles," WITH LIKES ",likes)


# ## LEAST LIKED VIDEO

# In[17]:


least_liked = df_details.sort_values(by='like_count', ascending=True).iloc[0]
least_liked[['video_id','title', 'like_count']]
titles = least_liked['title']
likes = least_liked['like_count']
print("LEAST LIKED VIDEO ",titles," WITH LIKES ",likes)


# ## MOST DURATION VIDEO

# In[18]:


# from the data took highest duriation viedo and converted to days 
highest_duration = df_details.sort_values(by='duration', key=lambda x: pd.to_timedelta(x),ascending=False).iloc[0]
highest_duration[['video_id','title' ,'duration']]
titles = highest_duration['title']
duration = highest_duration['duration']
print("MOST DURATION VIDEO ",titles," WITH  ", pd.to_timedelta(duration))


# ## Sentiment Analysis

# In[19]:


import re
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
sid = SentimentIntensityAnalyzer()


# In[20]:


# Get non empty comments , by filteration for sentiment analysis 
df_details = df_details[df_details['comments'].apply(lambda x: len(x) > 3)]
df_details.head(5)


# In[35]:


def preprocess_comments(comment_arr):
    cleaned_comments = []
   
    comment_arr=eval(comment_arr)
    
    stemmer = SnowballStemmer('english')
    for comment in comment_arr:
        if isinstance(comment, str):
            # Convert to lowercase
            comment = comment.lower()

            # Remove URLs
            comment = re.sub(r'http\S+', '', comment)
            
            # Remove usernames
            comment = re.sub(r'@\S+', '', comment)
            
            # Remove non-alphanumeric characters
            comment = re.sub('[^a-zA-Z0-9\s]', '', comment)
            
            # Tokenize
            tokens = nltk.word_tokenize(comment)
            
            # Remove stop words
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            
            # Stem words
            tokens = [stemmer.stem(token) for token in tokens]
            
            # Join tokens back into a single string
            comment = ' '.join(tokens)
            
            cleaned_comments.append(comment)
    return cleaned_comments


# In[36]:


df_details['comments_Processed'] = df_details['comments'].apply(lambda x: preprocess_comments(x))
df_details['comments_Processed']


# In[39]:


#calculate sentiment for all comments in a video and take average sentiment score for each video 

def get_sentiment(comment_list):
    
    sentiment_scores = []
    for comment in comment_list:
        
        if isinstance(comment, str):
            score = sid.polarity_scores(comment)
            sentiment_scores.append(score['compound'])
    return sum(sentiment_scores)/len(sentiment_scores)


# In[40]:


df_details['sentiment_score'] = df_details['comments_Processed'].apply(lambda x: get_sentiment(x))



# In[42]:


top10 = df_details.sort_values(by='sentiment_score', ascending=False).head(10)
top10[['video_id', 'title', 'sentiment_score']]

titles = top10['title']
sentiment_score = top10['sentiment_score']

# Create a bar diagram
plt.bar(titles, sentiment_score,)
plt.xticks(rotation=90)
plt.xlabel('Video Title')
plt.ylabel('sentiment_score')
plt.title('TOP 10 positive viewed Videos ')
plt.show()


# In[ ]:




