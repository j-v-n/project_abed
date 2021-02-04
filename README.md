# Project Abed

![Abed](https://i.gifer.com/8vg1.gif)

Project Abed is my first attempt at creating a chatbot with Reinforcement Learning!
This chatbot will be trained on reddit data. 
# Data
This project initially started when I followed the excellent Sentdex video series on creating [chatbots](https://www.youtube.com/playlist?list=PLQVvvaa0QuDdc2k5dwtDTyT9aCja0on8j). However, I wanted to create an RL based chatbot instead of using the base Google NMT model. The data is essentially the same : monthly reddit comment [data](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/?st=j9udbxta&sh=69e4fee7). I have used the months of January 2015, December 2016, February 2017 and January 2018. The total number of comments are over 7 million (99% came from the 2015 dataset. Somehow repeating the same process in the other datasets only produced a few thousand parent comment-reply pairs. I didn't bother further investigating why as I had asufficient number of comments parsed)

I stopped following along with the tutorial after video 7 in the series. For the rest of the process, I will be following the model building process described in Chapter 14 of Maxim Lapan's Deep Reinforcement Learning Hands-on book.

# Scripts
- utils/filter_data.py : Iterates through the from and to files to remove the " newlinechar " delimiter.  Instead of using the tokenizer Sentdex uses, I will be using NLTK for which this delimiter is unnecessary. 

# To-do
- Finish building data pipeline 
- Build model
- Train model
- ???
- Profit!
- Commit environment.yml file
