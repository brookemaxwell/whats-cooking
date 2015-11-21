import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, ElasticNet
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import tree
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import re
from sklearn import tree, cross_validation
from numpy import *
from sklearn.externals.six import StringIO
import random
import csv


def parse_recipe_file(filename, feature_vectorizer=None, label_set=None):
    recipes = json.load(open(filename))

    random.shuffle(recipes)

    raw_labels = []
    raw_features = []
    ids = []

    for recipe in recipes:
        if 'cuisine' in recipe.keys():
            raw_labels.append(recipe['cuisine'])
        else:
            ids.append(recipe['id'])
        ingredient_string = ' '.join(recipe['ingredients'])  # make comma separated string
        ingredient_string = re.sub('\(\w+\)', '', ingredient_string)  # delete quantities in parentheses
        ingredient_string = re.sub('\d+', '', ingredient_string)  # delete numbers
        raw_features.append(ingredient_string)

    if len(raw_labels) != 0:
        if label_set is None:
            label_set = list(set(raw_labels))

        labels = []
        for label in raw_labels:
            labels.append(label_set.index(label))

        return raw_features, labels, label_set
    else:
        return raw_features, ids


def split_test_and_training(features, labels, percent_training=0.7):
    num_training_instances = int(percent_training * len(labels))

    training_features = features[:num_training_instances]
    test_features = features[num_training_instances:]

    training_labels = labels[:num_training_instances]
    test_labels = labels[num_training_instances:]

    return training_features, training_labels, test_features, test_labels


def sgd():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('sgd', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
                         ])

    return text_clf


def naive_bayes():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())
                         ])

    return text_clf

def decision_tree():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('svd', TruncatedSVD(n_components=500)),
                         ('dt', tree.DecisionTreeClassifier())
    ])
    return text_clf

def voting_ensemble():
    vt = VotingClassifier(estimators=[('sgd1', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
                                      ('sgd2', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
                                      ('sgd3', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),

                                      #('mnb1', MultinomialNB()),
                                      #('mnb2', MultinomialNB(alpha=0.5)), #this hurts us
                                      #('svc', SVC()), # really really slow. No improvement
                                      # ('dt_stuff', decision_tree()),
                                      ('bnb1', BernoulliNB()),
                                      ('bnb2', BernoulliNB(alpha=0.5)),
                                      #('rf', ExtraTreesClassifier(n_estimators=200, max_features='auto', verbose=3, n_jobs=-1)),
                                      ('rf', ExtraTreesClassifier(n_estimators=200, max_features='auto', verbose=3, n_jobs=-1)),
                                      ('mlp', MLPClassifier(verbose=True))], voting='hard')

    eclf = Pipeline([('vect', CountVectorizer(binary=True, max_df=0.5)),
                     ('vote', vt)])
    #eclf = VotingClassifier(estimators=[('sgd', sgd()), ('nb', naive_bayes())], voting='hard')
    return eclf

def bagging_ensemble(estimator):
    """
    bag = BaggingClassifier(estimators=[('sgd1', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
                                      ('sgd2', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
                                      ('sgd3', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
                                      #('mnb1', MultinomialNB()),
                                      #('mnb2', MultinomialNB(alpha=0.5)), #this hurts us
                                      #('svc', SVC()), # really really slow. No improvement
                                      # ('dt_stuff', decision_tree()),
                                      ('bnb1', BernoulliNB()),
                                      ('bnb2', BernoulliNB(alpha=0.5))], voting='hard')
    """
    bag = BaggingClassifier(base_estimator=estimator, n_estimators=20, max_samples=0.4)

    eclf = Pipeline([('vect', CountVectorizer()),
                         #('tfidf', TfidfTransformer()), #frequency. removing improved. why?
                         ('bag', bag)])
    #eclf = VotingClassifier(estimators=[('sgd', sgd()), ('nb', naive_bayes())], voting='hard')
    return eclf

def ensemble_of_ultimate_glory():
    vt= VotingClassifier(estimators=[('ve', voting_ensemble()),
                                     #('be', bagging_ensemble(MultinomialNB())),
                                     ('be2', bagging_ensemble(BernoulliNB())),
                                     ('be3', bagging_ensemble(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)))], voting='hard')
    return vt

def random_forest():
    rf = Pipeline([('vect', CountVectorizer()),
                   ('rf', RandomForestClassifier(n_estimators=100, max_features='auto', verbose=3, n_jobs=-1))])

    return rf


def extra_random_forest():
    rf = Pipeline([('vect', CountVectorizer()),
                   ('rf', ExtraTreesClassifier(n_estimators=100, max_features='auto', verbose=3, n_jobs=-1))])

    return rf

def deep_learning():
    nn = Pipeline([('vect', CountVectorizer()),
                   ('mlp', MLPClassifier(verbose=True))])

    return nn


def try_local(filename, algorithm):
    features, labels, label_set = parse_recipe_file(filename)
    label_set = asarray(label_set)
    print("file_parsed")

    '''training_features, training_labels, test_features, test_labels = split_test_and_training(features, labels)

    algorithm.fit(training_features, training_labels)

    predicted = algorithm.predict(test_features)
    accuracy = mean(predicted == test_labels)

    print(metrics.classification_report(test_labels, predicted, target_names=label_set))'''
    scores = cross_validation.cross_val_score(algorithm, features, labels, cv=5, n_jobs=-1)
    print(scores)
    print(scores.mean())

    #print("Overall Accuracy: " + str(accuracy))

def try_kaggle(training_filename, test_filename, algorithm):

    training_features, training_labels, label_set = parse_recipe_file(training_filename)
    label_set = asarray(label_set)

    algorithm.fit(training_features, training_labels)

    test_features, test_ids = parse_recipe_file(test_filename)
    test_predicted = algorithm.predict(test_features)

    with open('kaggle_out.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(["id", "cuisine"])
        data = list(zip(test_ids, label_set[test_predicted]))
        writer.writerows(data)


def main():
    training_filename = 'Data/train.json'
    kaggle_test_filename = 'Data/test.json'

    text_clf = voting_ensemble()
    try_local(training_filename, text_clf)
    #try_kaggle(training_filename, kaggle_test_filename, text_clf)


if __name__ == '__main__':
    main()