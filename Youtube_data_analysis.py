

# # SANDEEP KOLAPINENI 


import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import requests
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


# ## Data Collection 



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


    
video_ids = ['Tgm9Y7aj0pU','Lam9Y7aj0pU']

# Apply fetch_video_details function to each id and drop null values
df_details = video_ids.apply(fetch_video_details).dropna()

# Create a DataFrame from the fetched details
df_details = pd.DataFrame(df_details.tolist(), columns=['video_id', 'title', 'description',
                                                        'view_count',
                                                        'like_count', 'dislike_count', 'comment_count', 'duration',
                                                        'comments'])

# Save the details to file
df_details.to_csv('fetched_video_details.csv', index=False,mode='a',header=False)




# load the data for analysis 
df_details = pd.read_csv('fetched_video_details.csv')
df_details.head()




df_details = df_details.drop_duplicates(subset='video_id', keep='first')



top10 = df_details.sort_values(by='view_count', ascending=False).head(10)
top10[['video_id', 'title', 'view_count']]




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


# ## MOST LIKED VIDEO




most_liked = df_details.sort_values(by='like_count', ascending=False).iloc[0]
most_liked[['video_id',  'title','like_count']]
titles = most_liked['title']
likes = most_liked['like_count']
print("MOST LIKED VIDEO ",titles," WITH LIKES ",likes)




# ## MOST DURATION VIDEO

# In[18]:


# from the data took highest duriation viedo and converted to days 
highest_duration = df_details.sort_values(by='duration', key=lambda x: pd.to_timedelta(x),ascending=False).iloc[0]
highest_duration[['video_id','title' ,'duration']]
titles = highest_duration['title']
duration = highest_duration['duration']
print("MOST DURATION VIDEO ",titles," WITH  ", pd.to_timedelta(duration))

