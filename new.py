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
        self.d1 = np.zeros(32)
        self.d2 = []
        self.d3 = []
        self.d4 = []
        self.d5 = []
        self.d6 = []
        self.d7 = []
        self.d8 = []

class p_match:
    def __init__(self, u1p,v1p,u2p,v2p, u1c,v1c,u2c,v2c):
        self.u1p = u1p
        self.v1p = v1p
        self.u2p = u2p
        self.v2p = v2p
        self.u1c = u1c
        self.v1c = v1c
        self.u2c = u2c
        self.v2c = v2c


def nonMaximumSuppression(I, dims, pos, c, maxima):
    # if pos== true, find local maximum, else - find local minimum

    w = dims[0]
    h = dims[1]
    M= np.zeros((dims[0], dims[1]), dtype=np.int32)
    n = 5
    margin = 5
    tau = 50
#try np here too
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

    it.d1[0] = (I_du[u-3][v-1])
    it.d1[1] = (I_dv[u-3][v-1])
    it.d1[2] = (I_du[u-3][v+1])
    it.d1[3] = (I_dv[u-3][v+1])
    it.d1[4] = (I_du[u-1][v-1])
    it.d1[5] = (I_dv[u-1][v-1])
    it.d1[6] = (I_du[u-1][v+1])
    it.d1[7] = (I_dv[u-1][v+1])
    it.d1[8] = (I_du[u+1][v-1])
    it.d1[9] = (I_dv[u+1][v-1])
    it.d1[10]= (I_du[u+1][v+1])
    it.d1[11]= (I_dv[u+1][v+1])
    it.d1[12]= (I_du[u+3][v-1])
    it.d1[13]= (I_dv[u+3][v-1])
    it.d1[14]= (I_du[u+3][v+1])
    it.d1[15]= (I_dv[u+3][v+1])
    it.d1[16]= (I_du[u-5][v-3])
    it.d1[17]= (I_dv[u-5][v-3])
    it.d1[18]= (I_du[u-5][v+3])
    it.d1[19]= (I_dv[u-5][v+3])
    it.d1[20]= (I_du[u+5][v-3])
    it.d1[21]= (I_dv[u+5][v-3])
    it.d1[22]= (I_du[u+5][v+3])
    it.d1[23]= (I_dv[u+5][v+3])
    it.d1[24]= (I_du[u-1][v-5])
    it.d1[25]= (I_dv[u-1][v-5])
    it.d1[26]= (I_du[u-1][v+5])
    it.d1[27]= (I_dv[u-1][v+5])
    it.d1[28]= (I_du[u+1][v-5])
    it.d1[29]= (I_dv[u+1][v-5])
    it.d1[30]= (I_du[u+1][v+5])
    it.d1[31]= (I_dv[u+1][v+5])


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
   # extract maxima via non-maximum suppression


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
def match_features(maxima1, i1, maxima2, i2, now_prev):
    M = 200

    min_sad1 = 1000000
    min_sad2 = 1000000

    # tmp_0 = abs(features_2_np_u - features1.u) < M
    # tmp_1 = abs(features_2_np_v - features1.v) < M
    # result = tmp_0 & tmp_1


    for i in range(0,len(maxima2)):  #try to make a vector of np, make a vector with u,v,c (probably)

        if abs(maxima2[i].u - maxima1[i1].u) < M and abs(maxima2[i].v - maxima1[i1].v) < M and (now_prev or abs(maxima2[i].v - maxima1[i1].v)<= 1):
            sad = SAD_NP(maxima1[i1].d1, maxima2[i].d1)
            if sad < min_sad1:
                min_sad1 = sad
                i2 = i

            elif sad < min_sad2:
                min_sad2 = sad

    if min_sad1< (min_sad2*0.95):
        return True, i2
    else:
        return False, None


def SAD_NP(descr1, descr2):
   return np.sum(np.abs(descr1-descr2))

def SAD(descr1,descr2):
  sad = 0
  for i in range (0, len(descr1)):
      sad += abs(descr1[i] - descr2[i])  #look for NP solution!

  return sad


def match_four_pictures(maxima1p, maxima2p, maxima1c, maxima2c):
    # u_max = 0
    # v_max = 0
    # for i in range(len(maxima1p)):
    #     if maxima1p[i].u > u_max:
    #         u_max = maxima1p[i].u
    #     if maxima1p[i].v > v_max:
    #         v_max = maxima1p[i].v
    #
    now = time.time()
    p = []
    i = 0
    files = 'tryingOCV_4pict_p3.dat'
    for i1p in range (len(maxima1p)):
        i2p, i1c, i2c, i1p_new = [0, 0, 0, 0]
        tmp, i2p =  match_features(maxima1p, i1p, maxima2p, i2p, False)
        if not tmp:
            continue
        tmp,i2c =  match_features(maxima2p, i2p, maxima2c, i2c, True)
        if not tmp:
            continue
        tmp, i1c = match_features(maxima2c, i2c, maxima1c, i1c, False)
        if not tmp:
            continue
        tmp, i1p_new = match_features(maxima1c, i1c, maxima1p, i1p_new, True)
        if not tmp:
            continue
        if i1p_new == i1p:
            u1p = maxima1p[i1p].u
            u2p = maxima2p[i2p].u
            u1c = maxima1c[i1c].u
            u2c = maxima2c[i2c].u
            v1p = maxima1p[i1p].v
            v2p = maxima2p[i2p].v
            v1c = maxima1c[i1c].v
            v2c = maxima2c[i2c].v

            p.append(p_match(u1p, v1p, u2p, v2p, u1c, v1c, u2c, v2c))
            f = open(files, 'a+b')
            pickle.dump(p[i], f)
            f.close()
            i+=1

    end = time.time()
    total = end-now
    print total
    return p


#if __name__ == '__main__':
#    global maxima, blob_img, corn_img

string0 = '/home/leyla/pyProjects/вапвап/libviso/img/000001_right.jpg'
string1 = '/home/leyla/pyProjects/вапвап/libviso/img/000001_left.jpg'

string_0r = '/home/leyla/pyProjects/вапвап/libviso/img/000002_right.jpg'
string_1l = '/home/leyla/pyProjects/вапвап/libviso/img/000002_left.jpg'

maxima0 =[]
maxima1 =[]

maxima_0r=[]
maxima_1l=[]

blob_img0=[]
corn_img0=[]
blob_img1=[]
corn_img1=[]

blob_img0r=[]
corn_img0r=[]
blob_img1l=[]
corn_img1l=[]

img0 = cv2.imread(string0, 0)
img1 = cv2.imread(string1, 0)
img0r = cv2.imread(string_0r, 0)
img1l = cv2.imread(string_1l, 0)

blob_img0, corn_img0 =findFeatures(img0, maxima0, blob_img0, corn_img0)  #right image
blob_img1, corn_img1 =findFeatures(img1, maxima1, blob_img1, corn_img1)  #left image

blob_img0r, corn_img0r =findFeatures(img0r, maxima_0r, blob_img0r, corn_img0r)
blob_img1l, corn_img1l =findFeatures(img1l, maxima_1l, blob_img1l, corn_img1l)
#print len(maxima0)
#print len(maxima1)

keypoints_0_0 =[]
keypoints_1_0 =[]

keypoints_0_r =[]
keypoints_1_l =[]

for i in range (0,len(maxima0)):
     # if maxima0[i].c == 0:
     keypoints_0_0.append(cv2.KeyPoint(y = maxima0[i].u, x = maxima0[i].v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))

for i in range (0,len(maxima1)):
     # if maxima1[i].c == 0:
     keypoints_1_0.append(cv2.KeyPoint(y = maxima1[i].u, x = maxima1[i].v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))

for i in range (0,len(maxima_0r)):
     # if maxima0[i].c == 0:
     keypoints_0_r.append(cv2.KeyPoint(y = maxima_0r[i].u, x = maxima_0r[i].v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))

for i in range (0,len(maxima_1l)):
     # if maxima1[i].c == 0:
     keypoints_1_l.append(cv2.KeyPoint(y = maxima_1l[i].u, x = maxima_1l[i].v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))


im_with_keypoints_0_0 = cv2.drawKeypoints(img0, keypoints_0_0, np.array([]), (0,0,255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints_1_0 = cv2.drawKeypoints(img1, keypoints_1_0, np.array([]), (0,0,255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

im_with_keypoints_0_r = cv2.drawKeypoints(img0r, keypoints_0_r, np.array([]), (0,0,255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints_1_l = cv2.drawKeypoints(img1l, keypoints_1_l, np.array([]), (0,0,255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# descr = []
#
keypoints_descr_0 = []
keypoints_descr_1 = []
keypoints_descr_0_r = []
keypoints_descr_1_l = []
descr = [] # match_features(maxima0, maxima1)

p = match_four_pictures(maxima1, maxima0, maxima_1l, maxima_0r)
# p = []
#
# files = 'tryingOCV_4pict_p2.dat'
# f = open(files, 'rb')
# print '1'
#
# i = 0
# while True:
#     try:
#         p.append( pickle.load(f))
#     except (EOFError):
#         break
# # f.close()
# print len(descr)
for i in range (0, len(p)):
 #    u1p,v1p,u2p,v2p,u1c,v1c,u2c,v2c = p[i]
     keypoints_descr_0.append(cv2.KeyPoint(y = p[i].u1p, x = p[i].v1p,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
     keypoints_descr_1.append(cv2.KeyPoint(y = p[i].u2p, x = p[i].v2p,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
     keypoints_descr_0_r.append(cv2.KeyPoint(y = p[i].u1c, x = p[i].v1c,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
     keypoints_descr_1_l.append(cv2.KeyPoint(y = p[i].u2c, x = p[i].v2c,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))

kpts_descr_0 = cv2.drawKeypoints(img0, keypoints_descr_0, np.array([]), (0,0,255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kpts_descr_1 = cv2.drawKeypoints(img1, keypoints_descr_1, np.array([]), (0,0,255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kpts_descr_0_r = cv2.drawKeypoints(img0r, keypoints_descr_0_r, np.array([]), (0,0,255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kpts_descr_1_l = cv2.drawKeypoints(img1l, keypoints_descr_1_l, np.array([]), (0,0,255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('imgl', img1)
cv2.imshow('imgr', img0)
#
# cv2.imshow('keypts_r', im_with_keypoints_0_0)
# cv2.imshow('keypts_1', im_with_keypoints_1_0)

cv2.imshow('features matched in 0th', kpts_descr_0)
cv2.imshow('features matched in 1st', kpts_descr_1)
cv2.imshow('features matched in 2nd', kpts_descr_0_r)
cv2.imshow('features matched in 3rd', kpts_descr_1_l)

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
