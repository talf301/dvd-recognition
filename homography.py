import cv2
import numpy as np
import scipy.spatial.distance
import cProfile
""" This file contains all functionality surrounding doing homography estimation. """

def do_matching(rdesc, tdesc):
    """ For each descriptor in rdesc, find the best matching descriptor in tdesc, and
    only include the ones satisfying Lowe's criteria (ratio of second to first best <0.7)
    :param rdesc: matrix of # of keypoints in ref x 128 for each feature in reference
    :param tdesc: matrix of # of keypoints in test x 128 with descriptors for each feature in test

    :return:
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
    :param rfeat: 4 x 2 matrix, values are x,y of each feature for reference matched to each feature in test in parallel
    :param tfeat: 4 x 2 matrix, values are x,y of each feature for test matched to each feature in reference in parallel

    :return:
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

    # Normalize so that h_33 = 1, since we can only get h up to a scale factor
    # TECHINCALLY this assumption can fail if h_33 = 0, but this should be an edge case we can reasonably ignore
    u, s, v = np.linalg.svd(A)
    return v[-1, :].reshape(3,3) / v[-1, -1]
    #return min_vec.reshape(3,3)



def estimate_homography(rdesc, rfeat, tdesc, tfeat, num_iters=1000, thresh=3.0):
    """ Estimate homography transformation from reference to test using RANSAC.
    :param rdesc: matrix of # of keypoints x 128 with descriptors for each feature in reference
    :param rfeat: matrix of # of keypoints x 2, values are x,y of each feature for reference
    :param tdesc: matrix of # of keypoints x 128 with descriptors for each feature in test
    :param tfeat: matrix of # of keypoints x 2, values are x,y of each feature for test
    :param num_iters: number of iterations of ransac to run
    :param thres: threshold for distance for a projection to be considered an inlier. Default is 3.0 as in opencv

    :return:
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
            # If ther eare no matches, stop
            if len(inds) == 0:
                return best_in, best_hom
            samples.append(np.random.choice(inds))
            inds = [x for x in inds if not tmat[x] == tmat[samples[j]]]

        # Compute the sampled homography
        hom = compute_homography(rfeat[rmat[samples], :], tfeat[tmat[samples], :])

        # Transform all of our points
        res_trans = cv2.perspectiveTransform(rfeat[rmat, :].reshape(-1, 1, 2), hom)
        # print res_trans.shape

        ref_xy = rfeat[rmat[0], :]
        res_trans1 = cv2.perspectiveTransform(ref_xy[:, None].reshape(-1, 1, 2), hom)
        # print res_trans1[0][0]
        # print res_trans[0,0,:]
        # Compute distances
        x = tfeat[tmat, :]
        y = res_trans[:, 0, :]
        # dists = np.diag(scipy.spatial.distance.cdist(tfeat[tmat, :], res_trans[:, 0, :]))
        dists = np.sqrt(np.power(x-y, 2).sum(axis=1))

        # Get inlier count
        inliers = sum(dists < thresh)


        # inliers = 0
        # for j in range(rmat.shape[0]):
        #     ref_xy = rfeat[rmat[j], :]
        #     test_xy = tfeat[tmat[j], :]
        #     # Compute transformed coordinates
        #     res_trans = cv2.perspectiveTransform(ref_xy[:, None].reshape(-1, 1, 2), hom)
        #     # print res_trans[0][0]
        #     if scipy.spatial.distance.cdist(test_xy[None, :], res_trans[0][0][None, :]) < thresh:
        #         inliers += 1

        # Check if best
        if inliers > best_in:
            best_in = inliers
            best_hom = hom

    return best_in, best_hom


def visualize_transformation(im, hom, height, width, thickness=5):
    """ Visualize the homography transformation by taking the dvd cover and localizing it on test image.

    :param im: test image to overlay bounding box on
    :param hom: 3x3 homography transformation matrix
    :param height: height of dvd cover
    :param width: width of dvd cover
    :param thickness: The thickness of the line to draw

    :return:
        im: The image with bounding box drawn on it
    """

    # Compute the corner points
    pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
    transf = cv2.perspectiveTransform(pts, hom)
    # Turn to int since we're going to be drawing lines
    transf = (transf + 0.5).astype(int)
    transf = transf[:, 0, :]

    # Add lines to image
    im = cv2.line(im, tuple(transf[0, :]), tuple(transf[1, :]), (255,0,0), thickness)
    im = cv2.line(im, tuple(transf[1, :]), tuple(transf[2, :]), (255,0,0), thickness)
    im = cv2.line(im, tuple(transf[2, :]), tuple(transf[3, :]), (255,0,0), thickness)
    im = cv2.line(im, tuple(transf[3, :]), tuple(transf[0, :]), (255,0,0), thickness)

    return im

if __name__ == '__main__':
    img = cv2.imread('test/image_01.jpeg')
    #img = cv2.imread('/Users/tal/Dropbox/School/Y4S1/CSC420/Assignment3/data/11.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread('DVDcovers/shrek2.jpg')
    #img2 = cv2.imread('/Users/tal/Dropbox/School/Y4S1/CSC420/Assignment3/data/toy.jpg')
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, tdesc = sift.detectAndCompute(gray, None)
    kp1, rdesc = sift.detectAndCompute(gray2, None)
    print rdesc.shape
    print len(kp1)
    print len(kp2)
    rfeat = np.array([k.pt for k in kp1])
    tfeat = np.array([k.pt for k in kp2])
    cProfile.run('estimate_homography(rdesc, rfeat, tdesc, tfeat, num_iters=100)')
    rmat, tmat = do_matching(rdesc, tdesc)
    print rmat.shape


    # visualize_transformation(img, hom, img2.shape[0], img2.shape[1], 'tes_visualization.jpeg')
