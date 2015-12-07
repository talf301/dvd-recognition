"""
Given a library of possible DVD covers, goes through a set of test images and tries to identify
which DVD is in each image, and then puts a bounding box around where the DVD is in the image.
"""

from homography import estimate_homography, visualize_transformation
from vocab_tree import VocabTree, Node

import cv2
import numpy as np

import os
import sys
import logging
from argparse import ArgumentParser


__author__ = 'Tal Friedman (talf301@gmail.com)'

def script(train_dir, test_dir, results_dir, save_vt, load_vt, depth):
    """ Run the main pipeline of this project

    Arguments are as described in parse_args
    """

    # If we have a stored model, load it
    if load_vt:
        vt = VocabTree.load_from_file(load_vt)
        logging.info('Loaded tree successfully! Tree has %d nodes' % vt.root.max_index)
    else:
        vt = VocabTree.build_from_directory(train_dir, L=depth)
        logging.info('Built tree successfully! Tree has %d nodes' % vt.root.max_index)

    # If we are saving the model, save it now
    if save_vt:
        vt.dump(save_vt)
        logging.info('Dumped tree into %s' % save_vt)

    # Now run the actual pipeline
    # Get test images
    test_images = [x for x in os.listdir(test_dir) if x.endswith('.jpg') or x.endswith('.jpeg')]

    # Build sift
    sift = cv2.xfeatures2d.SIFT_create()

    # Make the results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Go through test images
    for test_im in test_images:
        # Load and get features
        test_image = cv2.imread(test_dir + os.sep + test_im)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        kp, tdesc = sift.detectAndCompute(gray, None)
        tfeat = np.array([k.pt for k in kp])

        # Get most similar from tree
        most_similar = vt.get_most_similar(tdesc)

        # Now let's use ransac to get the best result
        best_im_name = ""
        best_im = None
        most_in = 0
        best_rdesc = None
        best_rfeat = None
        best_hom = None
        for train in most_similar:
            # Load and get features
            train_image = cv2.imread(train_dir + os.sep + train)
            gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
            kp, rdesc = sift.detectAndCompute(gray, None)
            rfeat = np.array([k.pt for k in kp])

            # Estimate
            inl, hom = estimate_homography(rdesc, rfeat, tdesc, tfeat)

            # Check if best
            if inl > most_in:
                most_in = inl
                best_hom = hom
                best_im_name = train
                best_rdesc = rdesc
                best_rfeat = rfeat
                best_im = train_image

        # If we didn't find any sort of match at all, just ignore this file
        if most_in == 0:
            continue

        # Now that we've found the best match, dump the results in results folder
        im = visualize_transformation(test_image, best_hom, best_im.shape[0], best_im.shape[1])
        im = cv2.putText(im, best_im_name, (5,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
        out_im = results_dir + os.sep + test_im
        cv2.imwrite(out_im, im)
        logging.info('Wrote a result to %s' % out_im)



def parse_args(args):
    """ Deal with parsing command line arguments

    :param args: arguments
    :return: properly parse arguments
    """
    parser = ArgumentParser(description=__doc__.strip())

    parser.add_argument('train_dir', metavar='TRAIN',
                        help='Directory where dvd cover images are located.')
    parser.add_argument('test_dir', metavar='TEST',
                        help='Directory where test images are located.')
    parser.add_argument('results_dir', metavar='RES',
                        help='Directory in which to dump results')
    parser.add_argument('--save_vt', dest='save_vt', default=None,
                        help='Dump the vocabulary tree into the given .pkl file')
    parser.add_argument('--load_vt', dest='load_vt', default=None,
                        help='Load a vocabulary tree from the given .pkl file')
    parser.add_argument('--depth', dest='depth', default=4,
                        help='The maximum depth for the vocabulary tree. Will be ignored if loading.')

    return parser.parse_args(args)

def main(args = sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=logging.INFO)
    script(**vars(args))

if __name__ == '__main__':
    sys.exit(main())