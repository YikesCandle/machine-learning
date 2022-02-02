import csv
import math
import numpy
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus


DIM = 4
MAX = [-1, -1, -1, -1]
DATA = [[]]

LAST_CLUSTER = 0


def fill_data():
    global DATA
    with open("data_customers.csv") as csv_data:
        data = csv.reader(csv_data, delimiter=';')
        DATA = [[0.0 for _ in range(DIM)] for _ in data]
    with open("data_customers.csv") as csv_data:
        data = csv.reader(csv_data, delimiter=';')
        for i, line in enumerate(data):
            print(line)
            for p in range(DIM):
                if line[p] in ["Male", "Female"]:
                    value = 1 if line[p] == "Male" else 0
                else:
                    value = int(line[p])
                MAX[p] = max(MAX[p], value)
                DATA[i][p] = value
            print(DATA[i])
    MAX[0] = 1


def get_distance(point_a, point_b):
    distance = 0
    for i in range(DIM):
        distance += ((point_a[i] - point_b[i]) / MAX[i]) ** 2
    return math.sqrt(distance)


def get_centroid(cluster):
    centroid = [0.0 for _ in range(DIM)]
    for point in cluster[0]:
        for i in range(DIM):
            centroid[i] += point[i] / MAX[i]
    for i in range(DIM):
        centroid[i] /= float(len(cluster))
    return centroid


def get_distance_of_clusters(cluster_a, cluster_b):
    return get_distance(get_centroid(cluster_a), get_centroid(cluster_b))


def save_csv(data):
    with open("cluster_customer.csv", "w") as f:
        writer = csv.writer(f)
        for i, cluster in enumerate(data):
            for point in cluster[0]:
                result = []
                for d in point:
                    result.append(d)
                result.append(i)
                writer.writerow(result)


def calculate_z(clusters):
    z = list()
    last_cluster = len(DATA) - 1
    while True:
        min_cluster_a_index = -1
        min_cluster_b_index = -1
        min_distance = float("inf")
        for i, cluster_a in enumerate(clusters):
            for j, cluster_b in enumerate(clusters):
                if j <= i:
                    continue
                dist = get_distance_of_clusters(cluster_a, cluster_b)
                if dist < min_distance:
                    min_distance = dist
                    min_cluster_a_index = i
                    min_cluster_b_index = j
        assert min_cluster_a_index != -1
        a_name = clusters[min_cluster_a_index][1]
        b_name = clusters[min_cluster_b_index][1]
        z.append([float(a_name), float(b_name), float(min_distance),
                  float(len(clusters[min_cluster_a_index][0]) + len(clusters[min_cluster_b_index][0]))])
        if len(clusters) == 8:
            save_csv(clusters)
            return z

        last_cluster += 1
        clusters[min_cluster_a_index][1] = last_cluster
        for i, point in enumerate(clusters[min_cluster_b_index][0]):
            clusters[min_cluster_a_index][0].append(point)
        del clusters[min_cluster_b_index]


col_names = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "label"]
pima = pd.read_csv("cluster_customer.csv", header=None, names=col_names)
feature_cols = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = pima[feature_cols]    # Features
y = pima.label                # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

#print accuracy and visualize
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=[str(int('0') + x) for x in range(8)])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('customers_rules_depthmax.png')
Image(graph.create_png())


#fill_data()
#START = [[[point], i] for i, point in enumerate(DATA)]
#tmp = calculate_z(START)
#print(tmp)
# Z = numpy.array(tmp, dtype=float)
# print(Z)
# plt.title("Customer Dendograms")
# dg = sch.dendrogram(Z)
# plt.axhline(y=0.03, c='k')
# plt.show()
# print(MAX)
