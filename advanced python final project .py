#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# import warnings filter
from warnings import simplefilter


def logicsticregression_accuracy(train_X, train_y, test_X, test_y):
    lg = LogisticRegression(multi_class='auto', solver='lbfgs',max_iter=10000)
    lg.fit(train_X, train_y)
    predict_lg = lg.predict(test_X)
    accuracy = accuracy_score(test_y, predict_lg)
    return accuracy


def linearsvc_accuracy(train_X, train_y, test_X, test_y):
    lsvc = LinearSVC(max_iter=1000,dual=False)
    lsvc.fit(train_X, train_y)
    predict_lsvc = lsvc.predict(test_X)
    accuracy = accuracy_score(test_y, predict_lsvc)
    return accuracy


def KNN_accuracy(train_X, train_y, test_X, test_y):
    knn = KNeighborsClassifier()
    knn.fit(train_X, train_y)
    predict_knn = knn.predict(test_X)
    accuracy = accuracy_score(test_y, predict_knn)
    return accuracy


def svc_accuracy(train_X, train_y, test_X, test_y):
    svc = SVC(max_iter=10000)
    svc.fit(train_X, train_y)
    predict_svc = svc.predict(test_X)
    accuracy = accuracy_score(test_y, predict_svc)
    return accuracy


def tree_accuracy(train_X, train_y, test_X, test_y):
    dtc = DecisionTreeClassifier()
    dtc.fit(train_X, train_y)
    predict_dtc = dtc.predict(test_X)
    accuracy = accuracy_score(test_y, predict_dtc)
    return accuracy


def SGD_accuracy(train_X, train_y, test_X, test_y):
    sgd = SGDClassifier()
    sgd.fit(train_X, train_y)
    predict_sgd = sgd.predict(test_X)
    accuracy = accuracy_score(test_y, predict_sgd)
    return accuracy


def hard_ensemble(train_X, train_y, test_X, test_y):
    # hard ensemble

    voting_clf = VotingClassifier(
        estimators=[
            ('logistic regression', LogisticRegression(multi_class='auto', solver='lbfgs',max_iter=10000)),
            ('svm', SVC()),
            ('LinearSVC', LinearSVC(dual=False)),
            ('KNN', KNeighborsClassifier()),
            ('Decisontree', DecisionTreeClassifier()),
            ('SGD', SGDClassifier()),
        ],
        voting='hard', n_jobs=1)

    voting_clf.fit(train_X, train_y)
    pred_voting = voting_clf.predict(test_X)
    accuracy = accuracy_score(test_y, pred_voting)
    return accuracy


def bagging_accuracy(train_X, train_y, test_X, test_y):
    bag_clf = BaggingClassifier(
        LogisticRegression(multi_class='auto', solver='lbfgs',max_iter=10000), n_estimators=100,
        max_samples=int(np.ceil(0.6 * train_X.shape[0])),
        bootstrap=True, n_jobs=3, random_state=42)
    bag_clf.fit(train_X, train_y)
    pred_bag = bag_clf.predict(test_X)
    accuracy = accuracy_score(test_y, pred_bag)
    return accuracy


def pasting_accuracy(train_X, train_y, test_X, test_y):
    bag_clf = BaggingClassifier(
        LogisticRegression(multi_class='auto', solver='lbfgs',max_iter=10000), n_estimators=100,
        max_samples=int(np.ceil(0.6 * train_X.shape[0])),
        bootstrap=False, n_jobs=3, random_state=42)
    bag_clf.fit(train_X, train_y)
    pred_bag = bag_clf.predict(test_X)
    accuracy = accuracy_score(test_y, pred_bag)
    return accuracy


def randomforest_accuracy(train_X, train_y, test_X, test_y):
    rfc = RandomForestClassifier()
    rfc.fit(train_X, train_y)
    predict_rfc = rfc.predict(test_X)
    accuracy = accuracy_score(test_y, predict_rfc)
    return accuracy


def gradientboosting_accuracy(train_X, train_y, test_X, test_y):
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_clf.fit(train_X, train_y)
    predict_gb = gb_clf.predict(test_X)
    accuracy = accuracy_score(test_y, predict_gb)
    return accuracy


def adaboost_accuracy(train_X, train_y, test_X, test_y):
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, algorithm='SAMME.R',
                                 learning_rate=0.5, random_state=42)
    ada_clf.fit(train_X, train_y)
    predict_ada = ada_clf.predict(test_X)
    accuracy = accuracy_score(test_y, predict_ada)
    return accuracy


def XGBoost_accuarcy(train_X, train_y,test_X,test_y):
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(train_X, train_y)
    predict_xgb_model = xgb_model.predict(test_X)
    accuracy = accuracy_score(test_y, predict_xgb_model)
    return accuracy


# prepocessing
def preprocess_text(text):
    stops = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens_no_stop = [w for w in tokens if w not in stops]
    tokens_no_punc = [w for w in tokens_no_stop if w not in string.punctuation]
    stemmer = PorterStemmer()
    token_stem = [stemmer.stem(w) for w in tokens_no_punc]

    return token_stem


def main(file):
    # read data

    facts = pd.read_csv(file, encoding='unicode_escape')

    # 14640 reviews, 20 columns
    # among them, unit_id, _last_judgment_at,airline_sentiment_gold(repetition of label), tweet_created, tweet_id, tweet_location could not be used
    # airline_sentiment is the label
    # number: trusted_judgements, airline_sentiment:confidence,negativereason:confidence, retweet_count,
    # category: negativereason, airline
    # text: text

    print(list(facts.columns.values))
    simplefilter(action='ignore', category=FutureWarning)

    # code without using the negation information
    # categorical feature only contains three features
    category_fea = ['_unit_state', 'airline', 'user_timezone']
    num_fea = ['trusted_judgements', 'airline_sentiment:confidence', 'negativereason:confidence', 'retweet_count']
    text_fea = ['text']

    # preprocessing
    # get one-hot encoder for category features
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    # one-hot encoded the category features
    without_neg_reason = pd.get_dummies(facts[category_fea])
    # scale the numerical features
    scalar = StandardScaler()
    without_neg_reason['retweet_count'] = scalar.fit_transform(facts['retweet_count'].values.reshape(-1, 1))
    without_neg_reason['_trusted_judgments'] = scalar.fit_transform(facts['_trusted_judgments'].values.reshape(-1, 1))
    # adding other numerical features
    # filling null value
    median1 = facts['negativereason:confidence'].median()
    facts['negativereason:confidence'] = facts['negativereason:confidence'].fillna(median1)
    median2 = facts['airline_sentiment:confidence'].median()
    facts['airline_sentiment:confidence'] = facts['airline_sentiment:confidence'].fillna(median2)
    without_neg_reason['negativereason:confidence'] = scalar.fit_transform(
        facts['negativereason:confidence'].values.reshape(-1, 1))
    without_neg_reason['airline_sentiment:confidence'] = scalar.fit_transform(
        facts['airline_sentiment:confidence'].values.reshape(-1, 1))
    # label encoding the label
    labelencoder = LabelEncoder()
    without_neg_reason['label'] = labelencoder.fit_transform(facts['airline_sentiment'])

    # split
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(without_neg_reason, test_size=0.2, random_state=42)

    # get train data and label for train
    train_X = train.drop(["label"], axis=1).values
    train_y = train["label"].values

    # get test data and label for test
    test_X = test.drop(["label"], axis=1).values
    test_y = test["label"].values

    # calulating baseline
    from collections import Counter
    count_class = Counter(train_y)
    most = max(count_class)

    class_base = np.zeros(len(train_X), dtype=np.int)
    class_base[:] = most

    # baseline accuracy
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    accuracy = accuracy_score(train_y, class_base)
    print('accuracy for baseline: ', accuracy)

    # evaluate models

    # logistic regression
    lg_accuracy = logicsticregression_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for logistic regression with no negation information : ', lg_accuracy)

    # linearSVC
    ls_accuracy = linearsvc_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for linear SVC with no negation information : ', ls_accuracy)

    # KNN
    k_accuracy = KNN_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for KNN with no negation information : ', k_accuracy)

    # SVC
    s_accuracy = svc_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for SVC with no negation information : ', s_accuracy)

    # treeclassifier
    t_accuracy = tree_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for Decision Tree Classifier with no negation information : ', t_accuracy)

    # SGD
    sgd_accuracy = SGD_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for SGD with no negation information : ', sgd_accuracy)

    ########################
    # adding negation information
    # categorical features has four features
    category_fea = ['_unit_state', 'airline', 'user_timezone', 'negativereason']
    num_fea = ['trusted_judgements', 'airline_sentiment:confidence', 'negativereason:confidence', 'retweet_count']
    text_fea = ['text']

    # see the content of negative reason
    print(facts['negativereason'].unique())

    # preprocessing
    # get one-hot encoder for category features
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    with_negation = pd.get_dummies(facts[category_fea])

    print(with_negation.shape)

    # ordinalencoder for labels (not for randomforest)
    labelencoder = LabelEncoder()
    with_negation['label'] = labelencoder.fit_transform(facts['airline_sentiment'])

    # stand standard numerical features
    scalar = StandardScaler()
    with_negation['retweet_count'] = scalar.fit_transform(facts['retweet_count'].values.reshape(-1, 1))
    with_negation['_trusted_judgments'] = scalar.fit_transform(facts['_trusted_judgments'].values.reshape(-1, 1))

    # adding more number features
    with_negation['negativereason:confidence'] = facts['negativereason:confidence']
    with_negation['airline_sentiment:confidence'] = facts['airline_sentiment:confidence']

    # filling null value
    median1 = with_negation['negativereason:confidence'].median()
    with_negation['negativereason:confidence'] = with_negation['negativereason:confidence'].fillna(median1)
    median2 = with_negation['airline_sentiment:confidence'].median()
    with_negation['airline_sentiment:confidence'] = with_negation['airline_sentiment:confidence'].fillna(median2)

    # split

    train, test = train_test_split(with_negation, test_size=0.2, random_state=42)

    # get train data and label for train
    train_X = train.drop(["label"], axis=1).values
    train_y = train["label"].values
    # get test data and label for test
    test_X = test.drop(["label"], axis=1).values
    test_y = test["label"].values

    from sklearn.metrics import classification_report

    # evaluate model again
    # logistic regression

    lg_accuracy = logicsticregression_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for logistic regression with  negation information : ', lg_accuracy)

    # linearSVC
    ls_accuracy = linearsvc_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for linear SVC with  negation information : ', ls_accuracy)

    # KNN
    k_accuracy = KNN_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for KNN with  negation information : ', k_accuracy)

    # SVC
    s_accuracy = svc_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for SVC with no negation information : ', s_accuracy)

    # treeclassifier
    t_accuracy = tree_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for Decision Tree Classifier with  negation information : ', t_accuracy)

    # SGD
    sgd_accuracy = SGD_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for SGD with negation information : ', sgd_accuracy)

    # hard ensemble
    hv_accuracy = hard_ensemble(train_X, train_y, test_X, test_y)
    print('accuracy for Hard ensemble with negation information : ', hv_accuracy)

    # soft ensemble -- bagging
    b_accuracy = bagging_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for bagging  with negation information : ', b_accuracy)

    # soft ensemble -- pasting
    p_accuracy = pasting_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for pasting  with negation information : ', p_accuracy)

    # ensemble random forest
    rf_accuracy = randomforest_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for random forest  with negation information : ', rf_accuracy)

    # boosting -- gradient boosting
    gb_accuracy = gradientboosting_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for gradient boosting with negation information : ', gb_accuracy)

    # boosting -- adaboost
    ada_accuracy = adaboost_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for ada boost with negation information : ', ada_accuracy)

    # boosting-- xgboost
    xgb_accuracy=XGBoost_accuarcy(train_X, train_y,test_X,test_y)
    print('accuracy for XG boost with negation information : ', xgb_accuracy)

    # try polynominal features
    from sklearn.preprocessing import PolynomialFeatures
    # degree=3
    poly = PolynomialFeatures(3)

    train_X = train.drop(["label"], axis=1)
    train_y = train["label"].values

    test_X = test.drop(["label"], axis=1)
    test_y = test["label"]

    poly_X_train = poly.fit_transform(train_X)
    poly_X_test = poly.transform(test_X)

    # logistic regression evaluation
    poly_lg_accuracy = logicsticregression_accuracy(poly_X_train, train_y, poly_X_test, test_y)
    print('accuracy for Logistic Regression with negation information and polynomial features : ', poly_lg_accuracy)

    # using the text feature only
    # tf-idf
    # without preprocessing
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    tfidfvector=TfidfVectorizer("content", lowercase=True, analyzer="word", token_pattern="\S+", use_idf=True,
                                     min_df=10)
    labelencoder = LabelEncoder()
    comments = facts['text'].tolist()
    label_list = facts['airline_sentiment'].tolist()
    label = labelencoder.fit_transform(label_list)
    X = tfidfvector.fit_transform(comments)
    data = X.toarray()
    train_X = data[:11712]
    train_y = label[:11712]
    test_X = data[11712:]
    test_y = label[11712:]
    tfidf_lg_accuracy_no = logicsticregression_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for Logistic Regression with TF-IDF without preprocessing  : ', tfidf_lg_accuracy_no)

    # Tf-idf
    # preprocessed

    preprocessed_texts = []
    for c in comments:
        preprocessed = preprocess_text(c)
        preprocessed = ' '.join(preprocessed)
        preprocessed_texts.append(preprocessed)

    X = tfidfvector.fit_transform(preprocessed_texts)

    data = X.toarray()

    train_X = data[:11712]
    train_y = label[:11712]
    test_X = data[11712:]
    test_y = label[11712:]

    tfidf_lg_accuracy = logicsticregression_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for Logistic Regression with TF-IDF with preprocessing  : ', tfidf_lg_accuracy)

    # Bag of words
    # no - preprocessing
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(lowercase=False,
                           ngram_range=(1, 2),
                           max_features=5000,
                           token_pattern=u"(?u)\\b\\S+\\b",
                           analyzer="word")
    X = vectorizer.fit_transform(comments)
    data = X.toarray()

    train_X = data[:11712]
    train_y = label[:11712]
    test_X = data[11712:]
    test_y = label[11712:]

    bow_lg_accuracy_no = logicsticregression_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for Logistic Regression with Bag of words without preprocessing  : ', bow_lg_accuracy_no)

    # processed

    X = vectorizer.fit_transform(preprocessed_texts)

    data = X.toarray()

    train_X = data[:11712]
    train_y = label[:11712]
    test_X = data[11712:]
    test_y = label[11712:]

    bow_lg_accuracy = logicsticregression_accuracy(train_X, train_y, test_X, test_y)
    print('accuracy for Logistic Regression with Bag of words with preprocessing  : ', bow_lg_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="Airline-Sentiment-2-w-AA.csv", help='data')
    args = parser.parse_args()
    main(args.file)
