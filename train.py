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
import pydot
import random


def parse_recipe_file(filename, feature_vectorizer=None, label_set=None):
    recipes = json.load(open(filename))

    random.shuffle(recipes)

    raw_labels = []
    raw_features = []

    for recipe in recipes:
        raw_labels.append(recipe['cuisine'])
        ingredient_string = ' '.join(recipe['ingredients'])  # make comma separated string
        ingredient_string = re.sub('\(\w+\)', '', ingredient_string)  # delete quantities in parentheses
        ingredient_string = re.sub('\d+', '', ingredient_string)  # delete numbers
        raw_features.append(ingredient_string)

    #if feature_vectorizer is None:
    #    feature_vectorizer = CountVectorizer()
    #    instances = feature_vectorizer.fit_transform(raw_features)
    #else:
    #    instances = feature_vectorizer.transform(raw_features)

    #print(feature_vectorizer.get_feature_names())
    #print(instances.toarray())

    if label_set is None:
        label_set = list(set(raw_labels))

    labels = []
    for label in raw_labels:
        labels.append(label_set.index(label))

    return raw_features, labels, label_set


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


def main():
    training_filename = 'Data/train.json'

    features, labels, label_set = parse_recipe_file(training_filename)
    print("file_parsed")

    training_features, training_labels, test_features, test_labels = split_test_and_training(features, labels)


    """
    countVectorizer = CountVectorizer()

    c_training_features = countVectorizer.fit_transform(training_features)


    tfidf = TfidfTransformer()
    tfidf_training_features = tfidf.fit_transform(c_training_features)

    clf = MultinomialNB().fit(tfidf_training_features, training_labels)


    c_test_features = countVectorizer.transform(test_features)

    tfidf_test_features = tfidf.transform(c_test_features)

    correct_count = 0
    for index, instance in enumerate(tfidf_test_features):
        if clf.predict(instance) == test_labels[index]:
            correct_count += 1


    accuracy = correct_count / double(len(test_labels))

    print(accuracy)
    """

    text_clf = sgd()

    text_clf.fit(training_features, training_labels)

    predicted = text_clf.predict(test_features)
    accuracy = mean(predicted == test_labels)

    print(accuracy)

    print(metrics.classification_report(test_labels, predicted, target_names=label_set))




    #dt = tree.DecisionTreeClassifier(random_state=0)
    #score = cross_val_score(dt, training_instances, training_labels, cv=10)
    # dt = dt.fit(training_instances.toarray(), training_labels)

    #print(score)


if __name__ == '__main__':
    main()