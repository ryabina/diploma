#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pickle
import time

#blob_img =[]
#corn_img =[]
class maximum:
    def __init__(self,u,v,val, c):
        self.u = u
        self.v = v
        self.val = val
        self.c = c
        self.d1 = []
        self.d2 = []
        self.d3 = []
        self.d4 = []
        self.d5 = []
        self.d6 = []
        self.d7 = []
        self.d8 = []


def nonMaximumSuppression(I, dims, pos, c, maxima):
    # if pos== true, find local maximum, else - find local minimum

    w = dims[0]
    h = dims[1]
    M = np.zeros((dims[0], dims[1]), dtype=np.int32)
    n = 5
    margin = 5
    tau = 50

    for i in range(n + margin, w - n - margin, n + 1):
        #    print "1 entered 1st cycle i=", i
        for j in range(n + margin, h - n - margin, n + 1):
            a = False
            #        print '2 entered 2 cycle, j=', j
            mi = i
            mj = j
            mval = I[i][j]

            for i2 in range(i, i + n + 1):
                for j2 in range(j, j + n + 1):
                    cval = I[i2][j2]
                    if pos:
                        if cval > mval:  # if pos find minimum else - maximum
                            mi = i2
                            mj = j2
                            mval = cval
                    else:
                        if cval < mval:
                            mi = i2
                            mj = j2
                            mval = cval

            for i2 in range(mi - n, min(mi + n, w - 1) + 1):
                for j2 in range(mj - n, min(mj + n, h - 1) + 1):
                    if i2 < i or i2 > i + n or j2 < j or j2 > j + n:
                        cval = I[i2][j2]
                        if pos:
                            if cval > mval:
                                a = True
                                break
                            else:
                                pass
                        else:
                            if cval<mval:
                                a = True
                                break
                if a:
                    break

            if pos:
                if mval >= tau and M[mi][mj] == 0:
                    maxima.append(maximum(mi, mj, mval, c))
                    M[mi][mj] == 1
            else:
                if mval <= tau and M[mi][mj] == 0:
                    maxima.append(maximum(mi, mj, mval, c))
                    M[mi][mj] == 1
                    #  goto should land right here and take us to another j
                    #   pass


def computeDescriptor(I_du, I_dv, u,v, it):
    it.d1.append(I_du[u-3][v-1])
    it.d1.append(I_dv[u-3][v-1])
    it.d1.append(I_du[u-3][v+1])
    it.d1.append(I_dv[u-3][v+1])
    it.d1.append(I_du[u-1][v-1])
    it.d1.append(I_dv[u-1][v-1])
    it.d1.append(I_du[u-1][v+1])
    it.d1.append(I_dv[u-1][v+1])
    it.d1.append(I_du[u+1][v-1])
    it.d1.append(I_dv[u+1][v-1])
    it.d1.append(I_du[u+1][v+1])
    it.d1.append(I_dv[u+1][v+1])
    it.d1.append(I_du[u+3][v-1])
    it.d1.append(I_dv[u+3][v-1])
    it.d1.append(I_du[u+3][v+1])
    it.d1.append(I_dv[u+3][v+1])
    it.d1.append(I_du[u-5][v-3])
    it.d1.append(I_dv[u-5][v-3])
    it.d1.append(I_du[u-5][v+3])
    it.d1.append(I_dv[u-5][v+3])
    it.d1.append(I_du[u+5][v-3])
    it.d1.append(I_dv[u+5][v-3])
    it.d1.append(I_du[u+5][v+3])
    it.d1.append(I_dv[u+5][v+3])
    it.d1.append(I_du[u-1][v-5])
    it.d1.append(I_dv[u-1][v-5])
    it.d1.append(I_du[u-1][v+5])
    it.d1.append(I_dv[u-1][v+5])
    it.d1.append(I_du[u+1][v-5])
    it.d1.append(I_dv[u+1][v-5])
    it.d1.append(I_du[u+1][v+5])
    it.d1.append(I_dv[u+1][v+5])


def computeDescriptors(I, dims, maxima):
    width = dims[0]
    height = dims[1]
    I_du = cv2.Sobel(I,cv2.CV_16S,1,0,ksize = 5, scale = 1, delta = 0,borderType = cv2.BORDER_DEFAULT)
    I_dv = cv2.Sobel(I,cv2.CV_16S,0,1,ksize = 5, scale = 1, delta = 0,borderType = cv2.BORDER_DEFAULT)

    for it in maxima:
        u = it.u
        v = it.v
        computeDescriptor(I_du,I_dv, u,v,it)

def findFeatures(img, maxima, blob_img, corn_img):

    kernel = np.array([[-1, -1, 0, 1, 1],
                       [-1, -1, 0, 1, 1],
                       [ 0,  0, 0, 0, 0],
                       [ 1,  1, 0,-1,-1],
                       [ 1,  1, 0,-1,-1]], np.float32)

    kernel2 = np.array([[-1,-1,-1,-1, -1],
                        [-1, 1, 1, 1, -1],
                        [-1, 1, 8, 1, -1],
                        [-1, 1, 1, 1, -1],
                        [-1,-1,-1,-1, -1]], np.float32)

    blob_img = cv2.filter2D(img, -1, kernel2)
    corn_img = cv2.filter2D(img, -1, kernel)

    dims = img.shape[0:2]  # dims[0] - width, dims[1] = height
   # extract maxima via /home/leyla/PycharmProjects/untitled/tryingOpenCv.pynon-maximum suppression


    nonMaximumSuppression(blob_img, dims, True, 0, maxima)
    nonMaximumSuppression(blob_img, dims, False, 1, maxima)
    nonMaximumSuppression(corn_img, dims, True, 2, maxima)
    nonMaximumSuppression(corn_img,dims, False, 3,maxima)
  #  print dims
  #  filter with sobel
  #  compute descriptor
    computeDescriptors(img, dims, maxima)
    num = len(maxima)
    if num == 0:
        max = 0
        return
    return blob_img, corn_img

###super not efficient  version of feature matching
def match_features(maxima1, maxima2):
    now = time.time()
    files = 'tryingOpenCv.datas2'
    descr = []
    i = 0
    M = 100
    features_1_np_u = np.zeros(len(maxima1))
    features_1_np_v = np.zeros(len(maxima1))
    for i in range (len(maxima1)):
        features_1_np_u[i] = maxima1[i].u
        features_1_np_v[i] = maxima1[i].v
    features_2_np_u = np.zeros(len(maxima2))
    features_2_np_v = np.zeros(len(maxima2))

    for i in range (len(maxima2)):
        features_2_np_u[i] = maxima2[i].u
        features_2_np_v[i] = maxima2[i].v

    for features1 in maxima1:
        feature = []
        min_sad1 = 1000000
        min_sad2 = 1000000
        
        for features2 in maxima2:  #try to make a vector of np, make a vector with u,v,c (probably)
        
            if abs(features2.u - features1.u) < M and abs(features2.v - features1.v) < M:
                sad = SAD(features1.d1, features2.d1)
                if sad < min_sad1:
                    min_sad1 = sad
                    feature = features2

                elif sad < min_sad2:
                    min_sad2 = sad

        if min_sad1< (min_sad2*0.9):
            descr.append([features1, feature])
            # f = open(files, 'a+b')
            # pickle.I_duq1mp(descr[i], f)
            # f.close()
            i += 1
    end = time.time()
    print end-now
    return descr

def SAD(descr1,descr2):
  t1 = time.time()
  sad = 0
  for i in range (0, len(descr1)):
      sad += abs(descr1[i] - descr2[i])  #look for NP solution!
  t2 = time.time()
  t = t2 - t1
  return sad

#if __name__ == '__main__':
#    global maxima, blob_img, corn_img

string0 = '/home/leyla/pyProjects/вапвап/libviso/img/000001_right.jpg'
string1 = '/home/leyla/pyProjects/вапвап/libviso/img/000001_left.jpg'
maxima0 =[]
maxima1 =[]
blob_img0=[]
corn_img0=[]
blob_img1=[]
corn_img1=[]
img0 = cv2.imread(string0, 0)
img1 = cv2.imread(string1, 0)
blob_img0, corn_img0 =findFeatures(img0, maxima0, blob_img0, corn_img0)
blob_img1, corn_img1 =findFeatures(img1, maxima1, blob_img1, corn_img1)
#print len(maxima0)
#print len(maxima1)

keypoints_0_0 =[]
keypoints_0_1 =[]
keypoints_0_2 =[]
keypoints_0_3 =[]

keypoints_1_0 =[]
keypoints_1_1 =[]
keypoints_1_2 =[]
keypoints_1_3 =[]

for i in range (0,len(maxima0)):
     # if maxima0[i].c == 0:
     keypoints_0_0.append(cv2.KeyPoint(y = maxima0[i].u, x = maxima0[i].v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))

for i in range (0,len(maxima1)):
     # if maxima1[i].c == 0:
     keypoints_1_0.append(cv2.KeyPoint(y = maxima1[i].u, x = maxima1[i].v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))




im_with_keypoints_0_0 = cv2.drawKeypoints(img0, keypoints_0_0, np.array([]), (0,0,255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints_1_0 = cv2.drawKeypoints(img1, keypoints_1_0, np.array([]), (0,0,255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

descr = []
# descr = match_features(maxima0, maxima1)


# files = 'tryingOpenCv.datas2'
# f = open(files, 'rb')
# print '1'

# i = 0
# while True:
#     try:
#         descr.append( pickle.load(f))
#     except (EOFError):
#         break
# f.close()
# print len(descr)
keypoints_descr_0 = []
keypoints_descr_1 = []
#

# descr1 = []
descr = match_features(maxima0, maxima1)
# #
# #
# files1 = 'tryingOpenCv.datas'
# f1 = open(files1, 'rb')
#
# i = 0
# while True:
#     try:
#         descr1.append( pickle.load(f1))
#     except (EOFError):
#         break
# f1.close()
# print len(descr1)
#


for i in range (0, len(descr)):
     p0,p1 =  descr[i]
     keypoints_descr_0.append(cv2.KeyPoint(y = p0.u, x = p0.v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
     keypoints_descr_1.append(cv2.KeyPoint(y = p1.u, x = p1.v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
kpts_descr_0 = cv2.drawKeypoints(img0, keypoints_descr_0, np.array([]), (0,0,255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kpts_descr_1 = cv2.drawKeypoints(img1, keypoints_descr_1, np.array([]), (0,0,255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('imgl', img1)
cv2.imshow('imgr', img0)
#
# cv2.imshow('keypts_r', im_with_keypoints_0_0)
# cv2.imshow('keypts_1', im_with_keypoints_1_0)

cv2.imshow('features matched in 0th', kpts_descr_0)
cv2.imshow('features matched in 1st', kpts_descr_1)

# #cv2.imshow('keypts_1', im_with_keypoints_1)
# #cv2.imshow('keypts_0', im_with_keypoints_0)
# #cv2.imshow('keypts_2', im_with_keypoints_2)
# #cv2.imshow('keypts_3', im_with_keypoints_3)


both = np.vstack((kpts_descr_1,kpts_descr_0))
#both = np.vstack((img1,img0))
print img1.shape
print both.shape
for i in range (0, len(descr), 5):
   if descr[i][1].c == 0:
       cv2.line(both,(descr[i][1].v,descr[i][1].u),(descr[i][0].v,descr[i][0].u+196),(255,0,0),1)
   if descr[i][1].c == 1:
       cv2.line(both,(descr[i][1].v,descr[i][1].u),(descr[i][0].v,descr[i][0].u+196),(0,255,0),1)
   if descr[i][1].c == 2:
       cv2.line(both,(descr[i][1].v,descr[i][1].u),(descr[i][0].v,descr[i][0].u+196),(0,0,255),1)
   if descr[i][1].c == 2:
       cv2.line(both,(descr[i][1].v,descr[i][1].u),(descr[i][0].v,descr[i][0].u+196),(0,100,100),1)
cv2.imshow("test",both)



cv2.waitKey(0)
# cv2.destroyAllWindows()
