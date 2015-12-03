from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2
import numpy as np
import scipy.spatial.distance
import os
import pickle

class VocabTree:
    """
    The main wrapper class and access point for using the vocabulary tree. User should only deal directly
    with a VocabTree object, but most of the work will be done in the linked structure of Nodes.
    images: list of image names aligned to indices
    k: branching factor of tree
    L: maximum depth of the tree
    root: root node of linked structure
    """
    def __init__(self, descs, labels, images, k=10, L=5):
        """

        :param descs: # of descs x 128 matrix of descriptors
        :param labels: # of descs sized vector, aligned labels for which image index each descriptor comes from
        :param images: list of image names, aligned to image indices
        :param k: The branching factor of the tree
        :param L: The max depth of the tree
        """

        self.images = images
        self.k = k
        self.L = L
        # Construct the actual tree, form the root node
        self.root = Node(descs, labels, 0, 0, k, L, len(images))


    def get_image_score_vec(self, descs):
        """

        :param descs: # of descs x 128 matrix of descriptors in image being scored

        :return:
            score: normalized vector of length # of nodes in tree with scores
        """

        # Empty vector to dump scores in
        score = np.zeros(self.root.max_index + 1)

        # Score
        self.root.compute_score_vector(descs, score)

        return score


class Node:
    """
    A node in our vocab tree. Each node will hold its weight, index, children, and sklearn KMeans object
    weight: The weight of this node for scoring purposes
    index: index of node
    depth: how deep in the tree the node is
    centers: The centers of the clusters. kx128 matrix
    children: a list of k children nodes, parallel to the first axis of centers
    max_index: the max index this node or any of its children has
    """
    def __init__(self, descs, labels, index, depth, k, L, db_im_size):
        """

        :param descs: # of descriptors x 128 matrix, rows parallel with labels, descriptors that go into this node
        :param labels: # of descriptors x,  aligned labels for which image index each descriptor comes from
        :param index: the index of this node (root should always be 0)
        :param depth: the depth in the tree of this node. Root has depth 0
        :param k: The branching factor of the tree
        :param L: The max depth of the tree
        :param db_im_size: The number of images in the database

        """
        if index % 1000 == 0:
            print index
            print depth
            print len(labels)
        self.index = index
        self.depth = depth

        # Compute the weight of this node as ln(N/N_i)
        n_i = len(set(labels))
        self.weight = np.log(float(db_im_size) / float(n_i))

        # If we've reached the depth cap, or there aren't enough descriptors, this is a leaf node and there's no work
        if depth == L or labels.shape[0] < k:
            self.centers = None
            self.max_index = index
            return

        # Compute our next split
        # If we have a lot of data, do mini batch for speed
        if len(labels) > 10000:
            kmns = MiniBatchKMeans(n_clusters=k)
            branches = kmns.fit_predict(descs)
        else:
            kmns = KMeans(n_clusters=k)
            branches = kmns.fit_predict(descs)

        # Store the centers
        self.centers = kmns.cluster_centers_

        # List of children
        self.children = []

        # Recursively create nodes
        for i in range(k):
            # Don't want to use the same index again
            index += 1
            # Get the descriptors and corresponding labels belonging to this branch
            branch_descs = descs[branches==i, :]
            branch_labels = labels[branches==i]
            # Create the new node
            branch = Node(branch_descs, branch_labels, index, depth + 1, k, L, db_im_size)
            self.children.append(branch)
            # Keep track of our index properly
            index = branch.max_index

        self.max_index = index


    def compute_score_vector(self, descs, scores):
        """
        Compute the score vector for a query document with descriptors descs, as described in Nister et Al.
        Nothing is returned, but at the end scores will be populated

        :param descs: # of descriptors x 128 matrix of descriptors to score
        :param scores: total # of nodes in tree sized vector with scores so we don't do lots of copying
        """

        # Add the score for the current node
        scores[self.index] += descs.shape[0] * self.weight

        # If this is a leaf node, stop
        if self.centers == None:
            return

        # Figure out how to split the data
        dists = scipy.spatial.distance.cdist(descs, self.centers)
        labels = np.argmax(dists, axis=1)

        # Get scores for children
        for i, n in enumerate(self.children):
            n.compute_score_vector(descs[labels == i, :], scores)

if __name__ == '__main__':
    sift = cv2.xfeatures2d.SIFT_create()
    train_dir = 'DVDcovers'
    # images = [x for x in os.listdir(train_dir) if x.endswith('.jpg')]
    # labels = []
    # descs = np.zeros((0, 128))
    # for i, image in enumerate(images):
    #     img = cv2.imread(train_dir + os.sep + image)
    #     print img.shape
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     _, desc = sift.detectAndCompute(gray, None)
    #     descs = np.vstack((descs, desc))
    #     labels.extend([i] * desc.shape[0])
    #
    # labels = np.array(labels)
    # pickle.dump((descs, labels, images), open('siftstuff.pkl', 'wb'))
    descs, labels, images = pickle.load(open('siftstuff.pkl', 'rb'))
    print 'done loading!'
    vt = VocabTree(descs, labels, images, L=2)
    test_image = cv2.imread('test/image_01.jpeg')
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    _, desc = sift.detectAndCompute(gray, None)
    print vt.get_image_score_vec(desc)
    # print root.max_index
    print labels.shape
    print descs.shape
