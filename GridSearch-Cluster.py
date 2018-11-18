
def GridSearch_Cluster(method = "KMeans", df = X, min_clusters = 2, 
                       max_clusters = 10, num_predictors = len(X.columns), 
                       affinity="euclidean", linkage = "complete"):
    """
    Grid-Search and feature selection for two clustering methods: KMeans and 
    Agglomerative Clustering using silhouette score

    Args:
        method (function): "KMeans", or "Agglomerative clustering"
        df (dataframe): dataframe of predictors
        min_clusters (int): minimum number of clusters to search over
        max_clusters (int): maximum number of clusters to search over
        num_predictors (int): number of variables in predictor combination
    Returns:
        Message with maximum score (float) and number of clusters, combination 
        of predictors that generate maximum score. (list)
    """
    import itertools
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
    
  
    total_columns = len(X.columns)

    #dictionary to store silhouette scores, number of clusters and variables 
    #used, keyed by silhouette score
    silhouette_scores = {}

    #list of scores to calculate max(score) and key dictionary for values
    scores = []

    #count to store progress
    count = 2

    #iterate over number of predictors specified
    for r in range(2,num_predictors+1): 
        #take combinations of size r from predictors
        #print(f"PROGRESS: {count}/{num_predictors}")
        for comb in itertools.combinations(X.columns, r): 
            # use dataframe with only predictors in combination
            X1 = X[list(comb)]
            #iterate over cluster range specified for predictor combination  
            for num_clusters in range(min_clusters, max_clusters): 
                if method == "KMeans":
                    cluster = KMeans(n_clusters=num_clusters)
                if method == "AgglomerativeClustering":
                    cluster = AgglomerativeClustering(n_clusters=num_clusters, 
                                                      affinity = affinity, 
                                                      linkage = linkage)
                cluster.fit(X1)
                y_cluster = cluster.predict(X1)
                score = silhouette_score(X1,y_cluster) 
                scores.append(score) 
                silhouette_scores[score] = [num_clusters,comb]
        print(f"combination analysis done for {count} predictors")
        count += 1


    return("Max score: " + max(scores) 
            + "; Optimum number of clusters and best predictor combination: " 
            + silhouette_scores[max(scores)])



