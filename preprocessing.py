#!/usr/bin/env python
# coding: utf-8


# read data
import argparse
import numpy as np
import pandas as pd
#import xgboost as xgb
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from warnings import simplefilter

def main(file):
    facts = pd.read_csv(file, encoding='unicode_escape')

    # drop useless features
    facts = facts.drop(['_unit_id'], axis=1)
    facts = facts.drop(["_golden"], axis=1)
    facts = facts.drop(["_last_judgment_at"], axis=1)

    facts = facts.drop(["tweet_created"], axis=1)
    facts = facts.drop(["tweet_id"], axis=1)
    facts = facts.drop(["tweet_location"], axis=1)
    facts = facts.drop(["name"], axis=1)
    facts = facts.drop(["airline_sentiment_gold"], axis=1)

    # label encoding
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    labelencoder = LabelEncoder()
    facts['label'] = labelencoder.fit_transform(facts['airline_sentiment'])
    # check label
    print('2 is', labelencoder.inverse_transform([2]))
    print('1 is', labelencoder.inverse_transform([1]))
    print('0 is ', labelencoder.inverse_transform([0]))
    # drop label
    facts = facts.drop(["airline_sentiment"], axis=1)

    # all useful features
    category_fea = ['_unit_state', 'airline', 'user_timezone', 'negativereason', 'negativereason_gold']
    num_fea = ['_trusted_judgments', 'airline_sentiment:confidence', 'negativereason:confidence', 'retweet_count']
    text_fea = ['text']

    # fill all null values
    # fill numerical features with median
    median1 = facts['negativereason:confidence'].median()
    facts['negativereason:confidence'] = facts['negativereason:confidence'].fillna(median1)
    median2 = facts['airline_sentiment:confidence'].median()
    facts['airline_sentiment:confidence'] = facts['airline_sentiment:confidence'].fillna(median2)
    # fill categorical features with the most common words
    common = facts['user_timezone'].mode()
    facts['user_timezone'] = facts['user_timezone'].fillna('Eastern Time (US & Canada)')
    # no negation reasons, then fill nothing
    facts['negativereason'] = facts['negativereason'].fillna('none')
    facts['negativereason_gold'] = facts['negativereason_gold'].fillna('none')

    facts = facts.drop(["tweet_coord"], axis=1)

    # split
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(facts, test_size=0.2, random_state=42)
    simplefilter(action='ignore', category=FutureWarning)
    # bag of words
    cvec = CountVectorizer(lowercase=False,
                           ngram_range=(1, 2),
                           # vocabulary=whitelist,   # You can work with your own whitelist
                           max_features=5000,  # Or work with the top 1000 most frequent items, or...
                           token_pattern=u"(?u)\\b\\S+\\b",  # Use these settings if you want to keep punctuation
                           analyzer="word")

    cvec.fit(train['text'])

    # preprocessing
    num_fea = ['_trusted_judgments', 'airline_sentiment:confidence', 'negativereason:confidence', 'retweet_count']
    num_fea_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    category_fea = ['_unit_state', 'airline', 'user_timezone', 'negativereason']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # For SVM and similar
        # ('ordinal', OrdinalEncoder())  # For Trees/Gradient Boosting
    ])

    text_fea = ['text']
    text_transformer = ColumnTransformer(transformers=[
        ('count', cvec, "text"),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('sca', num_fea_transformer, num_fea),
            ('text', text_transformer, text_fea),
            ('cat', categorical_transformer, category_fea)
        ])

    # encoded train and test data
    encoded = preprocessor.fit_transform(train)
    X_test = preprocessor.transform(test)

    # evaluate models
    # XGBOOST
    from sklearn.metrics import classification_report

    # boost_model = xgb.XGBClassifier().fit(encoded, train['label'])
    # X_test = preprocessor.transform(test)
    # preds = boost_model.predict(X_test)

    # print('classification report for xgboost:\n', classification_report(test["label"], preds))

    # boosting
    from sklearn.ensemble import GradientBoostingClassifier
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_clf.fit(encoded, train['label'])
    predict_gb = gb_clf.predict(X_test)
    print('classification report for gradient boost:\n', classification_report(test["label"], predict_gb))

    # Logitsic regression
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression()
    lg.fit(encoded, train['label'])
    predict_lg = lg.predict(X_test)
    print('classification report for logistic regression :\n', classification_report(test["label"], predict_lg))

    # pasting
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=100,
        max_samples=int(np.ceil(0.6 * encoded.shape[0])),
        bootstrap=False, n_jobs=3, random_state=42)
    bag_clf.fit(encoded, train['label'])
    pred_bag = bag_clf.predict(X_test)
    print('classification report for pasting:\n', classification_report(test["label"], pred_bag))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="Airline-Sentiment-2-w-AA.csv", help='data')
    args = parser.parse_args()
    main(args.file)
