from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2
import numpy as np
import scipy.spatial.distance
from collections import Counter
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
    norm_ord: The order of the norm to use - generally either L1 or L2
    db_scores: number of images in database x number of nodes in tree matrix of score vectors for each training image
    """
    def __init__(self, descs, labels, images, k=10, L=5, norm_ord=1):
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
        self.norm_ord = norm_ord

        # Empty list to populate for precomputing scores
        pre_scores = []
        # Construct the actual tree, from the root node
        self.root = Node(descs, labels, 0, 0, k, L, len(images), pre_scores)

        # Get our matrix of database scores
        self.db_scores = np.array(pre_scores).T

        # Nomralize database scores
        self.db_scores = self.db_scores / np.linalg.norm(self.db_scores, ord=norm_ord, axis=1)[:, None]

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

        # Normalize
        score = score / np.linalg.norm(score, ord=self.norm_ord)

        return score

    def get_most_similar(self, descs, n=10):
        """
        Gets the top n database images most similar to the query image described by descs

        :param descs: # of descriptors x 128 matrix of descriptors in query image
        :param n: top n hits will be returned

        :return:
            top_images: A list of the names of the top n matches
        """

        # Get our query score vector
        score_vec = self.get_image_score_vec(descs)

        # Compute normed distance from query to each db entry
        dists = np.linalg.norm(score_vec[None, :] - self.db_scores, ord=self.norm_ord, axis=1)

        # Get the ordering of distances, get top n
        ordering = np.argsort(dists)
        top_images = [self.images[ordering[i]] for i in range(n)]

        return top_images

    def dump(self, filename):
        """
        Dump this vocab tree into a file via pickling

        :param filename: Name of the file in which to dump
        """

        dump_file = open(filename, 'wb')
        pickle.dump(self, dump_file)
        dump_file.close()

    @staticmethod
    def load_from_file(filename):
        """
        Read in a previously dumped vocab tree from a pickled file

        :param filename: Name of the file from which to load

        :return:
            vt: The vocabulary tree read from file
        """

        load_file = open(filename, 'rb')
        vt = pickle.load(load_file)
        load_file.close()
        return vt


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
    def __init__(self, descs, labels, index, depth, k, L, db_im_size, pre_scores):
        """
        Create this node with given parameters, and recursively create children using k-means. You can
        think of this as being a root of a tree with height at most L-depth and branching factor k.

        :param descs: # of descriptors x 128 matrix, rows parallel with labels, descriptors that go into this node
        :param labels: # of descriptors x,  aligned labels for which image index each descriptor comes from
        :param index: the index of this node (root should always be 0)
        :param depth: the depth in the tree of this node. Root has depth 0
        :param k: The branching factor of the tree
        :param L: The max depth of the tree
        :param db_im_size: The number of images in the database
        :param pre_scores: List with a vector per node, with scores for each training image. Used for precomputing
            scores. Root should be given an empty list here.

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

        # First, assert that our indices haven't gotten out of order
        assert len(pre_scores) == index

        # Now compute this nodes vector of scores
        node_scores = np.zeros(db_im_size)
        counts = Counter(labels)
        for i in range(db_im_size):
            node_scores[i] += self.weight * counts[i]

        # Finally, add to the list
        pre_scores.append(node_scores)

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
            branch = Node(branch_descs, branch_labels, index, depth + 1, k, L, db_im_size, pre_scores)
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
        labels = np.argmin(dists, axis=1)

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
    # print min(labels)
    # print max(labels)
    print 'done loading!'
    vt = VocabTree(descs, labels, images, L=2)
    vt.dump('vt.pkl')
    vt = VocabTree.load_from_file('vt.pkl')
    test_image = cv2.imread('test/image_01.jpeg')
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    _, desc = sift.detectAndCompute(gray, None)
    print vt.get_most_similar(desc)
    # first_image = cv2.imread('DVDcovers/' + images[0])
    # gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    # _, desc = sift.detectAndCompute(gray, None)
    print vt.db_scores.shape
    # print root.max_index
    # print labels.shape
    # print descs.shape
