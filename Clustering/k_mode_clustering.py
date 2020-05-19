from kmodes.kmodes import KModes

cost = []
K = range(1,5)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(train_final)
    cost.append(kmode.cost_)
    
plt.plot(K, cost, 'bx-')
plt.xlabel('k clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()


km = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
cluster_labels = km.fit_predict(train_final)
train['Cluster'] = cluster_labels