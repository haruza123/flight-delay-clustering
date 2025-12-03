import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

st.title("Clustering Flight Delays (K-Means vs K-Medoids)")

uploaded = st.file_uploader("Upload Dataset (.csv/.xlsx)")

if uploaded:
    # Load data
    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Ambil kolom numerik
    numeric_cols = df.select_dtypes(include="number").columns
    st.write("Kolom numerik yang digunakan:", numeric_cols.tolist())

    data = df[numeric_cols].dropna()

    # Standardisasi
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    # Slider cluster
    k = st.slider("Jumlah Cluster (k)", 2, 10, 3)

    # ======================
    # K-MEANS
    # ======================
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    sil_kmeans = silhouette_score(X, kmeans_labels)

    # ======================
    # K-MEDOIDS (PyClustering)
    # ======================
    metric = distance_metric(type_metric.EUCLIDEAN)

    # Ambil k titik awal pertama
    initial_medoids = list(range(k))

    kmedoids_instance = kmedoids(X, initial_medoids, metric=metric)
    kmedoids_instance.process()

    clusters = kmedoids_instance.get_clusters()

    # Konversi cluster list → label array
    kmedoids_labels = [-1] * len(X)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            kmedoids_labels[idx] = cluster_id

    sil_kmedoids = silhouette_score(X, kmedoids_labels)

    # ======================
    # OUTPUT SILHOUETTE SCORE
    # ======================
    st.subheader("Silhouette Score")
    st.write("K-Means =", sil_kmeans)
    st.write("K-Medoids =", sil_kmedoids)

    # ======================
    # VISUALISASI PCA
    # ======================
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    st.subheader("Visualization")

    fig1 = plt.figure(figsize=(6,4))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmeans_labels, palette="Set2")
    plt.title("K-Means PCA")
    st.pyplot(fig1)

    fig2 = plt.figure(figsize=(6,4))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmedoids_labels, palette="Set1")
    plt.title("K-Medoids PCA")
    st.pyplot(fig2)
