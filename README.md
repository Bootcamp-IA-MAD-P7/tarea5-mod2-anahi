#Unsupervised Learning. Clustering Algorithms

https://bootcamp-ia-mad-p7.github.io/tarea5-mod2-anahi/

## 1. What is unsupervised learning & what's a clustering algorithm? What's the difference with supervised learning and what's its purpose?

The main types of ML are supervised and unsupervised. In **supervised** learning, the model is trained with labeled data, where each input has a corresponding output.
Types of supervised learning:
Regression. Used to predict continuous values, learns how to connect input data to a specific number or value.
Classification. Used to predict categorical data, learns how to connect input data with the probability of belonging to different groups of categories.

In the case of **unsupervised** learning, it involves training the model with unlabeled data, which helps to uncover patterns, structures or relationships within the data without predefined outputs. Basically, the machine learns by discovering hidden structures within the data without being told what the correct output should be. It’s divided in 2 categories of algorithms:
Association rule learning: Used to find patterns and relationships between different items in a dataset, it looks for rules. Some common ones are: Apriori, Eclat & FP-Growth.
Dimensionality reduction: Transform data from high-dimensional spaces to low-dimensional spaces without compromising meaningful properties in the original data.
Clustering.

### Applications

Unsupervised learning can be used for:
Natural language processing (NPL).
Image and video analysis.
Anomaly detection. Used to identify data points, events and/or observations that deviate from a dataset’s normal behavior.
Customer segmentation.
Recommendation engines.

### Clustering Algorithms

They’re used to group similar data points together without using labelled data that helps to discover hidden patterns or natural groupings in datasets. They work by repeatedly moving data points closer to the center of their group (cluster) and further from points in other groups. This helps the algorithm to create clear and meaningful clusters.
Types of clustering:
Hard clustering: Assigns each data point to exactly one cluster. A data point can’t belong to multiple clusters, there’s no overlap. Making the grouping clear and easy to interpret. Common uses: Market segmentation, customer grouping, document clustering.
Its limitation is that it can’t represent overlapping groups,it cannot handle situations where a data point may logically belong to multiple groups.
Soft clustering: It allows a data point to belong to multiple clusters with different probabilities. Instead of assigning a strict cluster, it gives a **degree** of membership to each cluster.
Common uses: Overlapping class boundaries, customer personas (multiple behavioral groups), medical diagnosis.
They’re beneficial to capture ambiguity and model gradual transitions.

In summary, untagged data is grouped based on their similarities and differences, to help identify groups with similar properties.

#### Main difference

Supervised learning → **External** evaluation. 
The answer key lives outside the model. The evaluation signal is **independent** of what the model learned.
Very clean measurable way.

Unsupervised learning → **Internal** evaluation
There’s no answer key. The evaluation is inherently self-referential: the model defines the structure, and the metrics validate that structure using the same data that generated it. There’s no equivalent of generalization error in the supervised sense, instead, you proxy it for using cluster stability across bootstrap samples or random initializations, and ultimately rely on downstream task performance or domain expert interpretation to determine whether the discovered structure is semantically meaningful, not just geometrically consistent.
The  model creates its own structure based entirely on the geometry of the input data.
It’s part math, part interpretation. Different metrics can differ. The result could be mathematically valid, but semantically meaningless. 
**Stability** becomes a primary validation tool → Running the same algorithm multiple times with different random seeds or on bootstrap samples of the data, and checking whether the clusters stay consistent.


## 2. Explain the conceptual difference between centroid-based (partitional) clustering and hierarchical clustering, mentioning one representative algorithm for each approach. 

The conceptual difference: Centroid requires to commit to a number of clusters in advance, before seeing the structure; while hierarchical doesn’t require to specify this upfront, it reveals the structure first and lets you decide where to cut after. The tradeoff is **cost**, centroid scales well to large datasets, and hierarchical is computationally expensive and becomes impractical at scale, because it needs to compute distances at every merge step. 

#### Centroid clustering

Groups data points around central points (centroids or medoids). Each cluster is represented by a point, and data points are assigned to the nearest centre. 

Algorithms: K-Means, K-Medroids (robust to outliers).
Advantages:Fast and scalable for large datasets, simple to implement and interpret.
Limitations: Requires choosing a number of clusters in advance, non suitable to non-spherical clusters.

#### Hierarchical clustering

Focuses on building a hierarchy of clusters in the form of a tree, also known as **Dendogram**, that shows relationships between clusters.

Approaches: Agglomerative (merges them step by step), Divisive (splits into smaller).
Advantages:Provides a full hierarchy, easy to visualize, not specifying number upfront.
Limitations: Computationally intensive for large datasets, merging/splitting decisions are **irreversible**, choosing parameters can be difficult.


## 3. What is density-based clustering (such as DBSCAN) and how does its approach differ from centroid-based and hierarchical clustering?

Instead of grouping, it defines clusters as regions of the feature space where datapoints are densely packed together. A point belongs to a cluster if it has enough neighbouring points within a given radius. Points in low density areas are treated as noise and not assigned.

Algorithms: DBSCAN (epsilon, min_samples), OPTICS.
Advantages: Handles clusters of varying shapes and sizes, doesn’t require cluster count upfront, effective in noisy datasets.
Limitations: Difficult to choose parameters like epsilon and min points, less effective for varying density clusters (except OPTICS).

The main divergence from connectivity based (hierarchical) and centroid is that those ones are compelled to assign every single point to a cluster, neither one has a concept of rejection. Basically density-based handles outliers, while the others don’t.


## 4. Why is it important to choose the number of clusters or the hyperparameters of a clustering algorithm correctly? Explain at least two methods for evaluating the quality of a clustering (for example, the elbow method and the silhouette coefficient).

## Number of clusters

**k** : Is the most critical hyperparameter in partitional clustering, because it defines the answer you’re looking for before it starts. It influences the clustering results. Too many can overfit the data, while not enough can miss important patterns.

## Hyperparameters

Clustering algorithms will always produce output regardless if the structure they find is real. Every algorithm has hyperparameters that directly define what it considers structure, they adjust how the algorithm behaves. But if their definition is incorrect the algorithm will return an invalid partition with no warning. 

**Distance metric** : Euclidean is the default in most algorithms. The choice of this metric changes what close means. Manhattan works better for high-dimensional data; cosine, for text. 

#### Kmeans Hyperparameters

**Initialization Method** : k-means++ (to improve convergence). Good initialization can lead to faster convergence and better clustering.
**n_init** : Number of initialization runs (n of times the algorithm will be run with different centroid initializations). Multiple initializations can prevent sub-optimal solutions, but increase computational cost.
**max_iter** : Maximum number of iterations the algorithm will run for each initialization. A higher number allows more time for convergence, but increases computational time.
**tol** : Tolerance to declare convergence. A smaller tolerance can lead to a more precise solution, but might increase computation time. Larger tolerance speeds up, but can lead to a less precise clustering.

### Methods for evaluating the quality:

**Elbow Method** : It works by training K-Means for a range of *k* values, recording inertia at each k, and plotting the curve.
Inertia decreases as k increases, more clusters mean smaller tighter groups, but the rate of decrease slows down past a certain point. That inflection point, where the curve bends from steep to flat is the “elbow”, and the *candidate k*, because it marks where adding another cluster stops absorbing genuinely  distinctive structure and start splitting already tight groups for *marginal gain* .
* Inertia is the sum of squared distances from each point to its assigned centroid across all clusters, it measures how compact the clusters are internally (k=1, big inertia, k=n of  points, inertia zero). 
The fundamental limitation is it's heuristic, not a formal mathematical rule, which makes it ambiguous often, that’s why it’s almost always used alongside the silhouette coefficient
**Silhouette coefficient** :  For each point measures how similar it is to its own cluster ( *cohesion* ) vs the nearest other cluster ( *separation* ). It returns a score from -1 to 1. Near 1 means well-placed, near 0 means on a boundary; negative means probably assigned to the *wrong cluster* . Works for any clustering algorithm, not only k-Means, and gives a single comparable number across different k values or algorithms.
**Davies-Bouldin Index** : Measures the average ratio of within-cluster scatter to between-cluster separation. Lower is better. It’s fully internal like silhouette, works for any algorithm, cheaper to compute.


## 5. Explain briefly the following algorithms

**K-Means** : It’s a hard clustering algorithm, which means there are no common points between 2 or more clusters. Partitions data into exactly k non-overlapping clusters, they can only belong to one cluster. It initializes k centroids, assigns each point to the nearest one, recomputes centroids as the mean of their assigned points, and repeats until assignments stop changing.

Advantages: Simple, fast, scales well to large datasets.
Limitations: Requires k upfront; assumes spherical equally-sized clusters; is sensitive to initialization and outliers; can converge to local optima.
Common use cases: Customer segmentation, image compression, document clustering.

**Agglomerative (Hierarchical)** : It starts by treating each data point as a separate cluster. After it calculates the distance between each pair of clusters (Euclidean, Manhattan, Cosine).  In each step, the algorithm merges the 2 clusters that are closest to each other (based on the distance matrix), after it recalculates the distances, updating the distance matrix (linkage criterion) until all the clusters are merged into one big cluster containing all data points. The result can be represented in a tree-like structure called _dendrogram, which shows the arrangement of the clusters and their proximity_.

Advantages: It doesn’t k upfront; reveals nested structure at multiple scales; deterministic.
Limitations: Can’t undo a merge once made; computational costs grow as the dataset n size grows, because it needs to recompute distances at every merge step.
Common use cases: Gene expression analysis, document taxonomy, social network community detection; data mining.

**DBSCAN** : It's a density-based clustering algorithm that divides the dataset into dense regions (clusters) separated by areas of low density. Clusters grow by connecting core points.
Its performance is sensitive to input parameters, like the radius of neighborhood and MinPts. Epsilon defines what "nearby" means; MinPts defines what "crowded enough" means. Dense enough regions become clusters, sparse regions become noise 

Advantages: It doesn’t require k upfront; finds arbitrarily shaped clusters; explicitly rejects noise points (outliers) instead of forcing them into a cluster.
Limitations: Sensitive to Epsilon and MinPts (bad hyperparameters produce completely wrong results); struggles with clusters of varying density.
Common use cases: Anomaly detection, geospatial analysis, image processing, network intrusion detection. Chosen when cluster shape is irregular and noise rejection matters

**Gaussian Mixture Models (GMM)** : It’s a soft (probability) clustering algorithm. It assumes the data was generated from a mixture of k Gaussian distributions, each with its own covariance and mean. Using Expectation-Maximization, estimates the parameters of those distributions and assigns each point a probability of belonging to each cluster, so a point can partially belong to multiple clusters simultaneously.

Advantages: Captures elliptical cluster shapes, not just spherical ones (k-means); provides uncertainty estimates per point; more flexible covariance structure.
Limitations: Sensitive to initialization; can converge to local optima; assumes data follows Gaussian distributions; computationally heavier than KMeans.
Common use cases: Anomaly detection, speaker recognition, image segmentation, medical imaging, financial modeling. Chosen when cluster boundaries are naturally fuzzy and you need probabilistic membership.

