# Segmentation
Symbol Segmentation (via Clustering) on CROHME dataset

S-oracle: every stroke is represented as a separate symbol, using the classification from the symbol that the stroke belongs to in ground truth.

## K-Means algorithm
Segment strokes using K-means algorithm over strokes, by at each iteration assigning strokes to the nearest of the k current cluster means using the distance between each mean and the average of the (x,y) coordinates belonging to a stroke. For each input formula, compute a separate k-means clustering for k∈{1,2, . . . , n}, where n is the number of strokes in the.inkml input.

We use the two different criterions below to selection the output from the candidate symbol segmentations (i.e., stroke sets) produced using k-means, one per value of k:
(a)k-oracle: for each file, selects the segmentation with the lowest value of k producing the largest number of correct symbols (i.e., in ground truth). All strokes receive their classification from ground truth.
(b)k-recognition: for each file, select the segmentation with the smallest number of symbols that produces the highest geometric mean over the class probabilities produced by the random forest classifier obtained from Classification part of https://github.com/anushreedas/Handwritten_Math_Symbol_Classification.

## Agglomerative clustering algorithm
At each iteration, merge the two clusters with the closest pair of sample points (Note:treat clusters as a set of (x,y) coordinates from strokes in the cluster). For each formula, keep track of the clustering tree (dendrogram) after each merge, so that you are able to evaluate the corresponding symbol segmentation at each step, beginning with n clusters (1 per stroke) and ending with one cluster holding all strokes.

We use each of the criterions below to select the final symbol segmentation taken from the agglomerative clustering tree for a file:
(a)ac-oracle:similar to k-oracle, for each file selects the segmentation with the lowest number of clusters producing the largest number of correct symbols from ground truth. All strokes receive their classification from ground truth.
(b)ac-recognition:for each file, select the segmentation with the smallest number of symbols that obtains the highest geometric mean over the classification probabilities from your random forest symbol classifier.

