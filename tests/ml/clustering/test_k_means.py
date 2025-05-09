from rooibos.ml.clustering.k_means import KMeans


def test_kmeans_initialization():
    model = KMeans(n_clusters=3, max_iter=100)
    assert model.n_clusters == 3
    assert model.max_iter == 100
    assert model.centroids is None


def test_kmeans_init_centroids():
    model = KMeans(n_clusters=2)
    centroids = model._init_centroids(dim=3)
    assert len(centroids) == 2
    assert all(len(c) == 3 for c in centroids)


def test_kmeans_train():
    model = KMeans(n_clusters=2, max_iter=10)
    X = [[1, 2], [2, 3], [10, 10], [11, 11]]
    centroids = model.train(X)
    assert len(centroids) == 2
    assert all(len(c) == 2 for c in centroids)


def test_kmeans_train_one_epoch():
    model = KMeans(n_clusters=2)
    X = [[1, 2], [2, 3], [10, 10], [11, 11]]
    initial_centroids = [[1, 2], [10, 10]]
    new_centroids = model.train_one_epoch(X, initial_centroids)
    assert len(new_centroids) == 2
    assert all(len(c) == 2 for c in new_centroids)


def test_kmeans_is_stable():
    model = KMeans(n_clusters=2)
    centroids1 = [[1.0, 2.0], [3.0, 4.0]]
    centroids2 = [[1.00001, 2.00001], [3.00001, 4.00001]]
    assert model.is_stable(centroids1, centroids2, tol=1e-4)


def test_kmeans_calculate_distance():
    model = KMeans(n_clusters=2)
    e = [1, 2]
    centroids = [[1, 2], [3, 4]]
    distances = model.calculate_distance(e, centroids)
    assert len(distances) == 2
    assert distances[0] == 0.0


def test_kmeans_calculate_clusters_for_dataset():
    model = KMeans(n_clusters=2)
    X = [[1, 2], [2, 3], [10, 10], [11, 11]]
    centroids = [[1, 2], [10, 10]]
    clusters = model.calculate_clusters_for_dataset(X, centroids)
    assert len(clusters) == 2
    assert len(clusters[0]) > 0
    assert len(clusters[1]) > 0