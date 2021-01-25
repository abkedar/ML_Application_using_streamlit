import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.title("Strealit Example")

st.write("""
# Explore different classifier
 Which one one is the best?
 """)

dataset_name = st.sidebar.selectbox("Select Dataset:", ('Iris', 'Breast cancer', 'Winequality_red'))

method = st.sidebar.selectbox("Select Correlation Method for Heat Map:", ('Spearman', 'Kendall', 'Pearson'))

classifier_name = st.sidebar.selectbox("Select Classifier:", ('KNN', 'SVM', 'Random Forest'))

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
        #df = pd.DataFrame(data)
    elif dataset_name == "Breast cancer":
        data = datasets.load_breast_cancer()
        #df = pd.DataFrame(data)
    else:
        data = datasets.load_wine()
        #df = pd.DataFrame(data)
    X = data.data
    y = data.target
    
    return X, y

X, y = get_dataset(dataset_name)

st.write("shape of the dataset", X.shape)
st.write("Number of class", len(np.unique(y)))

st.write("""
# Exploratory Data Analysis:
 """)



st.write("""
## **Heapmap Analysis Technique**
 """)

#df = pd.concat([X, y], axis=0)
st.write(pd.DataFrame(np.column_stack((X, y))).head())
df = pd.DataFrame(np.column_stack((X, y)))


def head_map(method, data):

    if method == 'Spearman':
        fig, ax = plt.subplots(figsize=(12,8))
        sns.heatmap(data.corr('spearman'), cmap="YlGnBu")
        st.pyplot(fig)
    elif method == 'Kendall':
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(data.corr('kendall'), cmap="YlGnBu")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(data.corr('pearson'), cmap="YlGnBu")
        st.pyplot(fig)

st.write(head_map(method, df))
st.write(dataset_name)


st.write("""
## Plotting
 """)

display = range(0, X.shape[1])
options = list(range(len(display)))
value = st.sidebar.selectbox("Seletec the nth number of Feature from dataset", options, format_func=lambda x: display[x])
st.write(value)

df_1 = pd.DataFrame(X)

#def plot_dist(dataset, data):
if dataset_name == 'Iris':
    hist_data = [df_1[0], df_1[1], df_1[2], df_1[3]]
    group_labels = [0, 1, 2, 3]#list(range(4))
    st.write(df.shape, len(df_1))
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 15, 20, 25])
    st.plotly_chart(fig, use_container_width=True)
elif dataset_name == 'Breast cancer':
    hist_data = [df_1[4], df_1[5], df_1[6], df_1[7], df_1[8], df_1[9], df_1[10], df_1[11], df_1[12], df_1[14], df_1[15], df_1[16],df_1[17], df_1[18], df_1[19], df_1[24], df_1[25], df_1[26], df_1[27], df_1[28], df_1[29]]
    group_labels = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27,28,29]#list(range(30))
    st.write(df.shape, len(df_1))
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 36, 38, 40, 42, 44])
    st.plotly_chart(fig, use_container_width=True)
else:
    hist_data = [df_1[0], df_1[1], df_1[2], df_1[3], df_1[5], df_1[6], df_1[7], df_1[8], df_1[9], df_1[10], df_1[11]]
    group_labels = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11 ]#list(range(13))
    st.write(df.shape, len(df_1))
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
    st.plotly_chart(fig, use_container_width=True)

# st.write(df.shape, len(df_1))
# fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 25])
# st.plotly_chart(fig, use_container_width=True)

st.write("""
# Predictive Modeling:
 """)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators 
    return params


params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                    max_depth=params["max_depth"], random_state=42)
    return clf

clf = get_classifier(classifier_name, params)


# Classification:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")


# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

# plt.show()
st.pyplot(fig)