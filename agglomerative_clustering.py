"""
@author:Sanket Agarwal
Python code for generating hierarchial agglomerative clustering of synthetic food data
"""

import math as m  # Importing  math library for math functions
from matplotlib import pyplot as plt  # Importing the Matplotlib library for drawing the graphs
from scipy.cluster.hierarchy import dendrogram, linkage  # Importing Dendogram and linkage libraries for making a dendogram
import csv  # Importing csv library


class ClusterPrototype:
    """
    Class for having all the information of a cluster (prototype)
    """
    sum_avg_attribute = None

    def __init__(self, data=None, label=None):
        """
        The constructor for initializing the cluster parameters
        :param data: The data
        :param label: Cluster label
        """
        self.data = data
        self.id_list = []  # list for storing the id of all the records in this cluster
        self.sum_avg_attribute = []  # storing the centre of mass for each attribute
        self.label = label
        if data != None:
            self.size = len(data)
            for col in range(1,len(self.data[0])):
                summ = 0
                for row in self.data:
                    summ += row[col]
                self.sum_avg_attribute.append(summ / self.size)
            for row in data:
                self.id_list.append(row[0])

    def merge_cluster(self, pro):
        """
        This method is called when we need to merge two clusters
        :param pro: The cluster which has to be merged with the calling cluster
        :return: None
        """
        for row in pro.data:
            self.data.append(row)
            self.id_list.append(row[0])
        self.size = len(self.data)
        self.sum_avg_attribute = []
        for col in range(1,len(self.data[0])):
            summ = 0
            for row in self.data:
                summ += row[col]
            self.sum_avg_attribute.append(summ / self.size)

    def get_data(self):
        """
        Get method to return data
        :return:  Data
        """
        return self.data

    def get_label(self):
        """
        Get method to return label
        :return: Label
        """
        return self.label

    def get_size(self):
        """
        Get method to return size of data
        :return: size
        """
        return self.size

    def get_id_list(self):
        """
        Get method to return id list
        :return: id list
        """
        return self.id_list

    def get_com(self):  # Center of mass for each attribute
        """
        Get method to return center of mass list
        :return: list of center of mass
        """
        return self.sum_avg_attribute


class HandlingData:
    """
    This class is mainly for handling the data and cluster creation
    """
    id_data_map = {}  # Id and data mapping
    data_list = []  # entire data
    clusters = []  # List for clusters
    merged_small_clusters = []  # list of the small clusters merged

    def __init__(self, file):
        """
        The constructor method for initializing cluster parameters
        :param file: The file
        """
        self.file = file
        with open(file) as f:
            count = 0
            for row in f:
                if count != 0:
                    self.data_list.append([int(_) for _ in row.split(',')])
                count += 1
            for row in self.data_list:
                self.id_data_map[row[0]] = row
        for key, row in self.id_data_map.items():  # Initial clustering
            self.clusters.append(ClusterPrototype([row], key))

    def calculate_euclidean_dist(self, com_1, com_2):
        """
        Method to calculate the euclidean distance between two clusters
        :param com_1:  The list of the center of mass of the cluster 1
        :param com_2:  The list of the center of mass of the cluster 2
        :return: Euclidean distance
        """

        squared_diff_sum = 0
        for d in range(len(com_1)):
            squared_diff_sum += (com_1[d] - com_2[d])**2

        return m.sqrt(squared_diff_sum)

    def cluster_creation(self):
            """
            Method for creating clusters
            :return: None
            """
            no_clusters__required = 3  # No. of clusters required
            cluster_1 = None
            cluster_2 = None
            cluster_1_ind = None
            cluster_2_ind = None


            while len(self.clusters) > no_clusters__required:
                min_distance = 99999
                prev_min_dist = 99999
                l = len(self.clusters)

                for i in range(len(self.clusters)-1):
                    for j in range(i+1,len(self.clusters)):
                        dist = self.calculate_euclidean_dist(self.clusters[i].get_com(), self.clusters[j].get_com())
                        if dist < min_distance:
                            min_distance = dist
                            cluster_1 = self.clusters[i]
                            cluster_2 = self.clusters[j]
                            cluster_1_ind = i
                            cluster_2_ind = j

                if prev_min_dist != min_distance:
                    if self.clusters[cluster_1_ind].get_label() <= self.clusters[cluster_2_ind].get_label():
                        if self.clusters[cluster_1_ind].get_size() <= self.clusters[cluster_2_ind].get_size():
                            self.merged_small_clusters.append(self.clusters[cluster_1_ind].get_size())
                        else:
                            self.merged_small_clusters.append(self.clusters[cluster_2_ind].get_size())
                        self.clusters[cluster_1_ind].merge_cluster(self.clusters[cluster_2_ind])

                        self.clusters.remove(self.clusters[cluster_2_ind])
                    else:
                        if self.clusters[cluster_1_ind].get_size() <= self.clusters[cluster_2_ind].get_size():
                            self.merged_small_clusters.append(self.clusters[cluster_1_ind].get_size())
                        else:
                            self.merged_small_clusters.append(self.clusters[cluster_2_ind])
                        self.clusters[cluster_2_ind].merge_cluster(self.clusters[cluster_1_ind].get_size())
                        self.clusters.remove(self.clusters[cluster_1_ind])
                    prev_min_dist = min_distance
            # Printing cluster informations
            self.clusters_info(self.clusters[0], "Cluster 1")
            self.clusters_info(self.clusters[1], "Cluster 2")
            self.clusters_info(self.clusters[2], "Cluster 3")

            # Saving the clusters in csv files
            self.make_csv(self.clusters[0], 'cluster1.csv')
            self.make_csv(self.clusters[1], 'cluster2.csv')
            self.make_csv(self.clusters[2], 'cluster3.csv')

            print("The size of smaller cluster that was merged in are: ")
            print(self.merged_small_clusters[-11:-1])

            # Drawing the dendogram
            self.draw_dendogram()

    def clusters_info(self, cluster, file):
        """
        Printing the cluster informations
        :param cluster: Cluster
        :param file: Name of the cluster
        :return: None
        """

        print("\n\n")
        print("Information for : "+ file)
        print("Cluster label: "+str(cluster.get_label()))
        data = cluster.get_data()
        '''for id in data:
            print(str(id) + ",")'''

        sum_avg_attribute = cluster.get_com()

        # header of the data (attributes)
        column_header = ['Milk', 'PetFood', 'Veggies', 'Cereal', 'Nuts', 'Rice', 'Meat', 'Eggs', 'Yogurt',
                         'Chips', 'Cola', 'Fruit']

        for i in range(len(sum_avg_attribute)):
            print(column_header[i]+": "+str(sum_avg_attribute[i]) + ",")
        '''sum = 0
        for row in data:
            sum += row[1]
        print("Milk avg: "+ str(sum/len(data)))'''

    def make_csv(self, cluster, file_name):
        """
        Method to make csv files
        :param cluster: Cluster
        :param file_name: Name of the file
        :return: NOne
        """
        column_header = ['ID', 'Milk', 'PetFood', 'Veggies', 'Cereal', 'Nuts'	,'Rice', 'Meat', 'Eggs', 'Yogurt', 'Chips',	'Cola',	'Fruit']
        data = cluster.get_data()
        with open(file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_header)
            writer.writeheader()
            for row in data:

                writer.writerow({column_header[0]: row[0], column_header[1]: row[1], column_header[2]: row[2],
                                 column_header[3]: row[3], column_header[4]: row[4], column_header[5]: row[5],
                                 column_header[6]: row[6], column_header[7]: row[7], column_header[8]: row[8],
                                 column_header[9]: row[9], column_header[10]: row[10], column_header[11]: row[11]
                                 ,column_header[12]: row[12]})

    def draw_dendogram(self):
        """
        method for drawing the dendogram
        :return: None
        """
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Index')
        plt.ylabel('Distance')
        Z = linkage(self.data_list, method='centroid')
        dendrogram(Z, truncate_mode='lastp', p=30)
        plt.show()


def main():
    """
    The main functin for calling the HandlingData() method
    :return: None
    """

    file_name = input("Enter the file name:")

    h = HandlingData(file_name)
    h.cluster_creation()

if __name__ == "__main__":
    main()






















