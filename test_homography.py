import cv2
import numpy as np
import homography

if __name__ == '__main__':
    img = cv2.imread('test/image_01.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img2 = cv2.imread('DVDcovers/shrek2.jpg')
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, tdesc = sift.detectAndCompute(gray, None)
    kp2, rdesc = sift.detectAndCompute(gray2, None)
    rfeat = np.array([k.pt for k in kp2])
    tfeat = np.array([k.pt for k in kp])
    inliers, hom = homography.estimate_homography(rdesc, rfeat, tdesc, tfeat, num_iters=1000)
    print inliers
    print hom

