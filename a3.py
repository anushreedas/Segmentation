"""
This program implements 5 segmentation systems for the strokes to get symbols
and stores the output.
The 5 systems are:
1. s-oracle
2. k-oracle
3. k-recognition
4. ac-oracle
5. ac-recognition

@author: Anushree Das
"""

import xml.etree.ElementTree as ET
from random import randrange
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import ExtractFeatures
import sys


class K_Means_Model:
    """
    Implements K-means clustering for given k
    """
    def __init__(self,k,max_epoch):
        self.k = k
        self.max_epoch = max_epoch

    def fit(self,strokes):
        self.centroids = []

        # select initial centroids
        for i in range(self.k):
            self.centroids.append(strokes[i].mean)

        for i in range(self.max_epoch):
            self.clusters = {}

            # initialize clusters
            for i in range(self.k):
                self.clusters[i] = []

            # add each stroke to a cluster whose centroid is closest to the mean of coordinates of the stroke
            for stroke in strokes:
                distances = []
                # find distance of the strokes mean to all centroids
                for centroid in self.centroids:
                    distances.append(np.linalg.norm(stroke.mean-centroid))
                # get index of the centroid which is closest
                classification = distances.index(min(distances))
                # add stroke to respective cluster
                self.clusters[classification].append(stroke)

            # store previous centroid
            centroids_old = self.centroids.copy()
            # calculate new centroid for each cluster
            for i in range(self.k):
                if self.clusters[i]:
                    self.centroids[i] = np.average([x.mean for x in self.clusters[i]], axis=0)

            optimized = True
            # check if the centroid moved fairly enough by comparing new centroid with the old one
            for i in range(self.k):
                original_centroid = centroids_old[i]
                current_centroid = self.centroids[i]
                error = np.linalg.norm(current_centroid - original_centroid)
                if error >= 0.1:
                    optimized = False

            if optimized:
                break


class K_Agglomerative_Model:
    """
    Stores clusters formed from agglomerative clustering gor a given k
    """
    def __init__(self,k,clusters):
        self.k = k
        self.clusters = clusters

    def get_info(self):
        """
        Returns list of list of coordinates of each stroke in each cluster
        """
        coods = []
        strokes = []
        for cluster in self.clusters:
            single_cluster_coods = []
            single_cluster_strokes = []
            for stroke in self.clusters[cluster]:
                single_cluster_coods.extend(stroke.coods)
                single_cluster_strokes.append(stroke)
            coods.append(single_cluster_coods)
            strokes.append(single_cluster_strokes)

        return coods, strokes


class Expression:
    def __init__(self, fname):
        self.filename = fname
        self.strokes = self.get_strokes()
        self.symbols_gt = []
        self.symbols = []
        self.k_means_clusters = []
        self.k_agglomerative_clusters = []

    def get_strokes(self):
        try:
            tree = ET.parse(self.filename, ET.XMLParser(encoding='utf-8'))
        except Exception:
            try:
                tree = ET.parse(self.filename, ET.XMLParser(encoding='iso-8859-1'))
            except Exception:
                # print('Error occured while reading inkml file:',Exception)
                return []
        # print('Getting strokes from inkml file..')
        root = tree.getroot()
        doc_namespace = "{http://www.w3.org/2003/InkML}"

        # extract stroke coordinates
        strokes = [Stroke(trace_tag.get('id'),
                          [[round(float(axis_coord))
                            if float(axis_coord).is_integer()
                            else round(float(axis_coord) * 10000)
                            for axis_coord in coord[1:].split(' ')]
                           if coord.startswith(' ')
                           else [round(float(axis_coord))
                                 if float(axis_coord).is_integer()
                                 else round(float(axis_coord) * 10000)
                                 for axis_coord in coord.split(' ')]
                           for coord in (trace_tag.text).replace('\n', '').split(',')])
                   for trace_tag in root.findall(doc_namespace + 'trace')]

        return strokes

    def set_symbol_gt(self):
        # print('Getting ground truth from inkml file..')
        try:
            tree = ET.parse(self.filename, ET.XMLParser(encoding='utf-8'))
        except Exception:
            try:
                tree = ET.parse(self.filename, ET.XMLParser(encoding='iso-8859-1'))
            except:
                return
        root = tree.getroot()
        doc_namespace = "{http://www.w3.org/2003/InkML}"

        # Always 1st traceGroup is a redundant wrapper'
        traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
        symbols = []

        if traceGroupWrapper is not None:
            # Process each symbol
            for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
                # get symbol class and id
                symbol_class = traceGroup.find(doc_namespace + 'annotation').text
                symbol_class = symbol_class.replace(',', 'COMMA')
                sym_id_ann = traceGroup.find(doc_namespace + 'annotationXML')

                if sym_id_ann:
                    sym_id = sym_id_ann.get('href')
                    sym_id = sym_id.replace(',', 'COMMA')
                else:
                    sym_id = symbol_class + '_' + str(randrange(100))

                # get stroke ids
                strokeid_list = []
                for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                    stroke_id = traceView.get('traceDataRef')
                    strokeid_list.append(stroke_id)
                # create Symbol object to store all ground truth extracted from the inkml file
                symbols.append(Symbol(sym_id, symbol_class, strokeid_list))

        self.symbols_gt = symbols

    def s_oracle(self):
        """
        In this system every stroke is represented as a separate symbol,
        using the classification from the symbol that the stroke belongs to in ground truth.
        """
        stroke_ids_all =[]
        # get ids of all strokes given in the inkml file
        if self.strokes:
            for stroke in self.strokes:
                stroke_ids_all.append(stroke.id)

        # get ground truth
        if len(self.symbols_gt) == 0:
            self.set_symbol_gt()

        # create separate Symbol object for each stroke id for each symbol in ground truth
        if self.symbols_gt:
            for symbol in self.symbols_gt:
                for stroke_id in symbol.stroke_list:
                    self.symbols.append(Symbol(symbol.symbol_id,symbol.symbol_class,[stroke_id]))
                    stroke_ids_all.remove(stroke_id)

        # create separate Symbol object for each stroke id which doesn't belong to any symbol
        for stroke_id in stroke_ids_all:
            self.symbols.append(Symbol('ABSENT_'+str(stroke_id), 'ABSENT', [stroke_id]))

        # write all output symbols created to .lg file
        self.write_lgfile('soracle')

    def find_min(self,list1,list2):
        """
        Returns minimum distance between all coordinates of two clusters
        """
        dist = []
        for cood1 in list1:
            for cood2 in list2:
                dist.append(np.linalg.norm(np.array(cood1)-np.array(cood2)))
        return min(dist)

    def agglomerative(self):
        # print('Running agglomerative')
        k_agglomerative_clusters = []
        for k in range(len(self.strokes),0,-1):
            clusters = {}
            # Initialize clusters
            for i in range(k):
                clusters[i] = []

            if k == len(self.strokes):
                # when k = number of strokes
                # each stroke will belong to a seperate cluster
                for i in range(len(self.strokes)):
                    clusters[i].append(self.strokes[i])
                k_agglomerative_clusters.append(K_Agglomerative_Model(k,clusters))
            else:
                coods, strokes = k_agglomerative_clusters[-1].get_info()

                # initialize distance matrix
                dist = [[np.Inf for _ in range(len(coods))] for _ in range(len(coods))]

                # get minimum distance between two clusters
                for i in range(len(coods)-1):
                    for j in range(i+1,len(coods)):
                        dist[i][j] = self.find_min(coods[i],coods[j])

                # find indices of clusters which are closest
                min_index = np.where(dist == np.amin(dist))
                closest_clusters = list(zip(min_index[0], min_index[1]))[0]

                index = 0
                # assign all old clusters which aren't needed to be merged to current clusters model
                for i in range(len(coods)):
                    if i not in closest_clusters:
                        clusters[index].extend(strokes[i])
                        index += 1
                # merge two clusters and add to current clusters model
                for i in closest_clusters:
                    for stroke in strokes[i]:
                        clusters[index].append(stroke)

                k_agglomerative_clusters.append(K_Agglomerative_Model(k, clusters))

        self.k_agglomerative_clusters = k_agglomerative_clusters

    def k_means(self):
        # print('Running Kmeans Algorithm')
        k_means_clusters = []
        # run k means algorithm for k = 1 to num of strokes
        if self.strokes :
            for k in range(1, len(self.strokes)+1):
                model = K_Means_Model(k, 100)
                model.fit(self.strokes)
                k_means_clusters.append(model)
            self.k_means_clusters = k_means_clusters

    def oracle(self,clustering_model):
        """
        Selects the segmentation with the lowest value of k
        producing the largest number of correct symbols (i.e., in ground truth).
        All strokes receive their classification from ground truth.
        """
        self.symbols = []
        if len(clustering_model) > 0:
            selected_k_index = 0
            max_correct = 0
            # get ground truth
            if len(self.symbols_gt) == 0:
                self.set_symbol_gt()

            for i in range(len(clustering_model)):
                model = clustering_model[i]
                correct = 0
                strokeid_list_gt = []

                # get clusters from ground truth
                for symbol in self.symbols_gt:
                    temp = sorted(symbol.stroke_list)
                    strokeid_list_gt.append(temp)

                # get clusters from model(kmeans/agglomerative)
                for cluster in model.clusters.keys():
                    strokeid_list = []
                    for stroke in model.clusters[cluster]:
                        strokeid_list.append(stroke.id)
                    # check if cluster is in ground truth for any of the symbol
                    if sorted(strokeid_list) in strokeid_list_gt:
                        correct += 1
                        strokeid_list_gt.remove(sorted(strokeid_list))

                # find model with max correct segmentation
                if correct > max_correct:
                    max_correct = correct
                    selected_k_index = i

            # Store the clusters for strokes for the selected k along with symbol class obtained from the ground truth
            for cluster in clustering_model[selected_k_index].clusters.keys():
                strokeid_list = []
                for stroke in clustering_model[selected_k_index].clusters[cluster]:
                    strokeid_list.append(stroke.id)
                found = False
                for symbol in self.symbols_gt:
                    if sorted(strokeid_list) == sorted(symbol.stroke_list):
                        self.symbols.append(symbol)
                        found = True
                if found == False and len(strokeid_list) > 0:
                    sym_class = 'unknown'
                    sym_id = sym_class+'_'+str(strokeid_list[0])
                    self.symbols.append(Symbol(sym_id,sym_class,strokeid_list))

    def ac_oracle(self):
        if len(self.k_agglomerative_clusters) == 0:
            self.agglomerative()
        self.oracle(self.k_agglomerative_clusters)
        self.write_lgfile('acoracle')

    def k_oracle(self):
        if len(self.k_means_clusters) == 0:
            self.k_means()
        self.oracle(self.k_means_clusters)
        self.write_lgfile('koracle')

    def geometric_mean(self,arr):
        """
        Calculates the geometric mean over the class probabilities
        """
        n = len(arr)
        prod = 1
        for row in arr:
            prod = prod * max(row)
        g_mean = prod ** (1 / n)
        return g_mean

    def recognition(self,clustering_model):
        """
        Select the segmentation with the smallest number of symbols
        that produces the highest geometric mean over the class probabilities
        produced by the random forest classifier from Project 1
        """
        self.symbols = []
        if len(clustering_model) > 0:
            # load random forest classifier
            rf_model = open('rf.pkl', 'rb')
            classifier = pickle.load(rf_model)

            selected_k_index = 0
            highest_gmean = 0

            for i in range(len(clustering_model)):
                model = clustering_model[i]
                features = []

                # extract features for each cluster
                for cluster in model.clusters.keys():
                    strokes = []
                    for stroke in model.clusters[cluster]:
                        strokes.append(stroke.coods)
                    if len(strokes) > 0:
                        features.append(ExtractFeatures.generate_features(strokes))

                # calculate class probabilities for each cluster
                class_probabilities = classifier.predict_proba(features)
                # get geometric mean
                g_mean = self.geometric_mean(class_probabilities)

                # select k with highest geometric mean
                if g_mean > highest_gmean:
                    selected_k_index = i
                    highest_gmean = g_mean

            # Store the clusters for strokes for the selected k
            # along with symbol class obtained from the classifier
            for cluster in clustering_model[selected_k_index].clusters.keys():
                strokes = []
                strokeid_list = []
                for stroke in clustering_model[selected_k_index].clusters[cluster]:
                    strokes.append(stroke.coods)
                    strokeid_list.append(stroke.id)
                if len(strokes) > 0:
                    # extract feature for cluster
                    features=ExtractFeatures.generate_features(strokes)
                    features = np.array(features).reshape(1, -1)
                    # predict class label for cluster
                    y_pred = classifier.predict(features)

                    sym_class = str(y_pred[0])
                    sym_id = sym_class + '_' + str(strokeid_list[0])
                    self.symbols.append(Symbol(sym_id, sym_class, strokeid_list))

    def k_recognition(self):
        if len(self.k_means_clusters) == 0:
            self.k_means()
        self.recognition(self.k_means_clusters)
        self.write_lgfile('krecognition_ex')

    def ac_recognition(self):
        if len(self.k_agglomerative_clusters) == 0:
            self.agglomerative()
        self.recognition(self.k_agglomerative_clusters)
        self.write_lgfile('acrecognition_test')

    def write_lgfile(self,directory):
        """
        writes the output to .lg file
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory,os.path.splitext(os.path.basename(self.filename))[0]+'.lg')
        with open(filepath,'w') as f:
            for symbol in self.symbols:
                f.write('O, '+str(symbol.symbol_id)+', '+symbol.symbol_class+', 1.0')
                for stroke_id in symbol.stroke_list:
                    f.write(', '+str(stroke_id))
                f.write('\n')

# data structure to store stroke information
class Stroke:
    def __init__(self,id,coods):
        self.coods = [[row[0],row[1]] for row in coods]
        self.id = id
        self.mean = np.mean(coods, axis=0)

# data structure to store symbol information
class Symbol:
    def __init__(self,sym_id,sym_class,stroke_l):
        self.symbol_id = sym_id
        self.symbol_class = sym_class
        self.stroke_list = stroke_l


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 a3.py [path to directory] [soracle|koracle|krecognition|acoracle|acrecognition]')
        exit(0)
    path = sys.argv[1]
    segmenter = sys.argv[2]

    if not os.path.exists(path):
        print("Path doesn't exist")
        exit(0)

    segmenters = ['soracle', 'koracle', 'krecognition', 'acoracle', 'acrecognition']
    if segmenter not in segmenters:
        print('Incorrect parameter value: ', segmenter)
        print('Usage: python3 a3.py [path to directory] [soracle|koracle|krecognition|acoracle|acrecognition]')
        exit(0)

    filelist = []
    # get all inkml files from directory and sub-directories
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.inkml':
                filelist.append(os.path.join(root, file))

    for filepath in tqdm(filelist):
        # print(filepath)
        e = Expression(filepath)
        if segmenter == 'soracle':
            e.s_oracle( )
        elif segmenter == 'koracle':
            e.k_oracle()
        elif segmenter == 'krecognition':
            e.k_recognition()
        elif segmenter == 'acoracle':
            e.ac_oracle()
        elif segmenter == 'acrecognition':
            e.ac_recognition()




