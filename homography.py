import cv2
import numpy as np
import scipy.spatial.distance
""" This file contains all functionality surrounding doing homography estimation. """

def do_matching(rdesc, tdesc):
    """ For each descriptor in rdesc, find the best matching descriptor in tdesc, and
    only include the ones satisfying Lowe's criteria (ratio of second to first best <0.7)
    args:
        rdesc: matrix of # of keypoints x dimensionality of descriptor with descriptors for each feature in reference
        tdesc: matrix of # of keypoints x dimensionality of descriptor with descriptors for each feature in test

    returns:
        rmat: indices for matches in reference, parallel to test matches
        tmat: indices for matches in test, parallel to reference matches
    """
    # Compute distances
    dists = scipy.spatial.distance.cdist(rdesc, tdesc).T

    # Get the ordering in which distances would be sorted, per reference descriptor, sort distances
    I = np.argsort(dists, axis=0)
    B = np.sort(dists, axis=0)

    # Get ratios
    ratios = B[0, :] / B[1, :]

    # Find ratios < 0.7, and corresponding matches
    rmat = np.nonzero(ratios < 0.7)[0]
    tmat = I[0, rmat]

    return rmat, tmat


def compute_homography(rfeat, tfeat):
    """ Given 4 matches tfeat and rfeat, compute solution for homography transformation
    args:
        rfeat: 4 x 2 matrix, values are x,y of each feature for reference matched to each feature in test in parallel
        tfeat: 4 x 2 matrix, values are x,y of each feature for test matched to each feature in reference in parallel

    returns:
        hom: the resulting transformation computed
    """

    # We will compute the homography by computing A as on slide 36 of lecture 9, then taking the
    # eigenvector of A^T*A with the smallest eigenvalue

    # Construct A
    A = np.zeros((8,9))
    for i in range(4):
        xr = rfeat[i, 0]
        yr = rfeat[i, 1]
        xt = tfeat[i, 0]
        yt = tfeat[i, 1]
        A[2*i, 0] = xr
        A[2*i, 1] = yr
        A[2*i, 2] = 1
        A[2*i+1, 3] = xr
        A[2*i+1, 4] = yr
        A[2*i+1, 5] = 1
        A[2*i, 6] = -xr * xt
        A[2*i, 7] = -xt * yr
        A[2*i, 8] = -xt
        A[2*i+1, 6] = -yt * xr
        A[2*i+1, 7] = -yr * yt
        A[2*i+1, 8] = -yt

    # Compute eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(A.T.dot(A))
    min_val = np.argmin(vals)
    min_vec = vecs[:, min_val]

    # Normalize so that h_33 = 1, since we can only get h up to a scale factor
    # TECHINCALLY this assumption can fail if h_33 = 0, but this should be an edge case we can reasonably ignore
    min_vec = min_vec / min_vec[-1]

    return min_vec.reshape(3,3)



def estimate_homography(rdesc, rfeat, tdesc, tfeat, num_iters=100, thresh=3.0):
    """ Estimate homography transformation from reference to test using RANSAC.
    args:
        rdesc: matrix of # of keypoints x dimensionality of descriptor with descriptors for each feature in reference
        rfeat: matrix of # of keypoints x 2, values are x,y of each feature for reference
        tdesc: matrix of # of keypoints x dimensionality of descriptor with descriptors for each feature in test
        tfeat: matrix of # of keypoints x 2, values are x,y of each feature for test
        num_iters: number of iterations of ransac to run
        thres: threshold for distance for a projection to be considered an inlier

    returns:
        best_in: The number of inliers found from the best trnasformation
        best_hom: The actual homography transformation matrix
    """
    best_in = 0
    best_hom = np.zeros((3,3))

    # Get matches
    rmat, tmat = do_matching(rdesc, tdesc)

    for i in range(num_iters):
        # We want to sample such that our ref and test matches are both unique
        samples = []
        inds = list(range(len(tmat)))
        for j in range(4):
            samples.append(np.random.choice(inds))
            inds = [x for x in inds if not tmat[x] == tmat[samples[j]]]

        # Compute the sampled homography
        hom = compute_homography(rfeat[rmat[samples], :], tfeat[tmat[samples], :])

        # Count inliers
        inliers = 0
        for j in range(rmat.shape[0]):
            ref_xy = rfeat[rmat[j], :]
            test_xy = tfeat[tmat[j], :]
            # Compute transformed coordinates
            res_trans = hom.dot(np.array([ref_xy[0], ref_xy[1], 1]))
            if scipy.spatial.distance.cdist(test_xy[None, :], res_trans[None, :2]) < thresh:
                inliers += 1

        # Check if best
        if inliers > best_in:
            best_in = inliers
            best_hom = hom

    return best_in, best_hom

