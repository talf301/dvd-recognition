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
    inliers, hm = homography.estimate_homography(rdesc, rfeat, tdesc, tfeat, num_iters=1000)
    rmat, tmat = homography.do_matching(rdesc, tdesc)
    print rmat.shape
    #h, _ = cv2.findHomography(rfeat[rmat, :], tfeat[tmat, :])
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(rdesc,tdesc,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    print len(good)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    hom, matchesMask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = gray2.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,hm)
    print dst

    img1 = cv2.polylines(img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    cv2.imwrite('test_buitin.jpeg', img1)

    #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #               singlePointColor = None,
    #               matchesMask = matchesMask, # draw only inliers
    #               flags = 2)
    #cv2.drawMatches(img2,kp1,img1,kp2,good, img3,**draw_params)

    #cv2.imshow(img3)

    homography.visualize_transformation(img, hom, img2.shape[0], img2.shape[1], 'tes_visualization.jpeg')
    #print hom
    #print h
