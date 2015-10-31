import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import SparsePCA
from sklearn.cross_validation import cross_val_score
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

    if feature_vectorizer is None:
        feature_vectorizer = CountVectorizer()
        instances = feature_vectorizer.fit_transform(raw_features)
    else:
        instances = feature_vectorizer.transform(raw_features)

    #print(feature_vectorizer.get_feature_names())
    #print(instances.toarray())

    if label_set is None:
        label_set = list(set(raw_labels))

    labels = []
    for label in raw_labels:
        labels.append(label_set.index(label))

    return instances, labels, feature_vectorizer, label_set

def export_tree(decision_tree, name):
    dot_data = StringIO()
    tree.export_graphviz(decision_tree, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(name + '.pdf')

def main():
    training_filename = 'Data/train.json'

    training_instances, training_labels, feature_vectorizer, label_set = parse_recipe_file(training_filename)
    print("file parsed")

    """
    pca = SparsePCA(n_components=100)
    pca_training_instances = pca.fit_transform(training_instances.toarray())
    print("pca complete")
    """

    dt = tree.DecisionTreeClassifier(random_state=0)
    score = cross_val_score(dt, training_instances, training_labels, cv=10)
    # dt = dt.fit(training_instances.toarray(), training_labels)

    print(score)


if __name__ == '__main__':
    main()