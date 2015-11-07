import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import re
from sklearn import tree
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


def try_local(filename, algorithm):
    features, labels, label_set = parse_recipe_file(filename)
    label_set = asarray(label_set)
    print("file_parsed")

    training_features, training_labels, test_features, test_labels = split_test_and_training(features, labels)

    algorithm.fit(training_features, training_labels)

    predicted = algorithm.predict(test_features)
    accuracy = mean(predicted == test_labels)

    print(accuracy)

    print(metrics.classification_report(test_labels, predicted, target_names=label_set))


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

    text_clf = sgd()
    try_local(training_filename, text_clf)
    #try_kaggle(training_filename, kaggle_test_filename, text_clf)


if __name__ == '__main__':
    main()