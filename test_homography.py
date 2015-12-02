import cv2
import numpy as np
import homography

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
    rfeat = np.array([k.pt for k in kp1])
    tfeat = np.array([k.pt for k in kp2])
    inliers, hom = homography.estimate_homography(rdesc, rfeat, tdesc, tfeat, num_iters=1000)
    rmat, tmat = homography.do_matching(rdesc, tdesc)
    print rmat.shape


    homography.visualize_transformation(img, hom, img2.shape[0], img2.shape[1], 'tes_visualization.jpeg')
