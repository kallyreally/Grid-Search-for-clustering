def GridSearch_Cluster(method = "KMeans", df = X, min_clusters = 2, max_clusters = 10, num_columns = len(X.columns), affinity="euclidean", linkage = "complete"):
    """
    Grid-Search for three clustering methods: KMeans, --- and -- using silhouette score

    Args:
        method (function): KMeans, Agglomerative clustering or ---.
        df (dataframe): dataframe of predictors
        min_clusters (int): minimum number of clusters to search over
        max_clusters: maximum number of clusters to search over
        num_columns: number of columns to search over

    Returns:
        Message with maximum score and number of clusters, combination of
        predictors that generate maximum score.
    """
    from sklearn.metrics import silhouette_score
    import itertools
    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
    
  
    total_columns = len(X.columns)

    #dictionary to store silhouette scores, number of clusters and variables used, keyed by silohouette score
    silhouette_scores = {}
    #list of scores to calculate max(score) and key dictionary for values
    scores = []
    comb_count = 0
    count = 2
    for r in range(2,num_columns):

        #print(" %i percent done ---------------------: ", count)
        count += 1
        for comb in itertools.combinations(X.columns, r):
            comb_count += 1
            print("combinations completed", comb_count)
            X1 = X[list(comb)]   
            for number in range(min_clusters, max_clusters):
                if method == "KMeans":
                    cluster = KMeans(n_clusters=number)
                if method == "AgglomerativeClustering":
                    cluster = AgglomerativeClustering(n_clusters=number, affinity = affinity, linkage = linkage)
                cluster.fit(X1)
                y_cluster = cluster.predict(X1)
                score = silhouette_score(X1,y_cluster)
                scores.append(score)
                silhouette_scores[score] = [number,comb]

    return "Max score: ", max(scores), "\n Optimum number of clusters and best predictor combination: ", silhouette_scores[max(scores)]



