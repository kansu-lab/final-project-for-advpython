# final-project-for-advpython
final project for advpython

The folder contains four files. 
# Airline-Sentiment-2-w-AA.csv
This file is the dataset I used for this dataset. It is downloaded from 
https://data.world/socialmediadata/twitter-us-airline-sentiment. 
The dataset is about Twitter US Airline Sentiment.
It contains 14640 rows and 20 columns. 

# advanced python final project.py 
This file contains my experiment 1-3, including baseline of my project, 
accuracy results when the learners just learn from categorical features without
negation information, and numerical features. 
It also calculates the improvement of accuracy score when negation information is
added. 
Besides, it evaluates the usefulness of ensemble. 
In the end, it presents the model performance when tweet texts are
encoded with TF-IDF and Bag of words.
 
#preprocessing.py
Codes in the file present the model performance of classifiers when
all the features (numerical features, categorical features, negation information and 
text features) are included. 

to run the code model required:
sklearn
nltk
xgboost

No configurations are required to be modified to run the code. 




