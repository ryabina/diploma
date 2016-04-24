#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

from math import *
import cv2
import pickle
import time
import matplotlib.pyplot as plt
import plotly.plotly as py
py.sign_in('ryabina_scout', '8d8wtczxli')
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
    def __init__(self, uLp,vLp,uRp,vRp, cL,cR):
        self.uLp = uLp
        self.vLp = vLp
        self.uRp = uRp
        self.vRp = vRp
        self.cL = cL
        self.cR = cR

class p_match_four:
    def __init__(self, uLp,vLp,uRp,vRp, uLc, vLc, uRc, vRc, cLp,cRp, cLc,cRc):
        self.uLp = uLp
        self.vLp = vLp
        self.uRp = uRp
        self.vRp = vRp

        self.uLc = uLc
        self.vLc = vLc
        self.uRc = uRc
        self.vRc = vRc

        self.cLp = cLp
        self.cRp = cRp
        self.cLc = cLc
        self.cRc = cRc


def nonMaximumSuppression(I,I_temp, dims, pos, c, maxima):
    # if pos== true, find local maximum, else - find local minimum

    w = dims[1]
    h = dims[0]
    n = 5
    margin = 5
    tau = 20
#try np here too
    for i in range(n + margin, h - n - margin, n):
        #    print "1 entered 1st cycle i=", i
        for j in range(n + margin, w - n - margin, n):
            a = False
            window = I[i:i+n, j:j+n]

            if pos:
                mval = np.amax(window)
                mcoords = np.unravel_index(np.argmax(window), window.shape) + np.array((i,j))
            else:
                mval = np.amin(window)
                mcoords = np.unravel_index(np.argmax(window), window.shape) + np.array((i,j))

            window2 = I[mcoords[0]-n:  min(mcoords[0]+ n, h - 1), mcoords[1]-n:min(mcoords[1] + n, w - 1)]
            cval = np.amax(window2)
            if pos:
                if cval > mval:
                    a = True
                    continue
            else:
                if cval <mval:
                    a = True
                    continue

            if pos:
                if mval >= tau and  I_temp[mcoords[0], mcoords[1]] == 0:
             #if I_temp[mcoords[0], mcoords[1]] == 0:
                    maxima.append(maximum(mcoords[0],mcoords[1], mval, c))
                    I_temp [mcoords[0], mcoords[1]] = 1
                   # print mcoords, "mval= ", mval

            else:
               if mval <= tau and I_temp[mcoords[0], mcoords[1]] == 0:
              # if I_temp[mcoords[0], mcoords[1]] == 0:
                    maxima.append(maximum(mcoords[0],mcoords[1], mval, c))
                    I_temp [mcoords[0], mcoords[1]] = 1
    #                print mcoords, "mval= ", mval




def computeDescriptor(I_du, I_dv, u,v, it):
    d = np.zeros(32)
    d[0] = I_du[u-3,v-1]
    d[1] = I_dv[u-3,v-1]
    d[2] = I_du[u-3,v+1]
    d[3] = I_dv[u-3,v+1]
    d[4] = I_du[u-1,v-1]
    d[5] = I_dv[u-1,v-1]
    d[6] = I_du[u-1,v+1]
    d[7] = I_dv[u-1,v+1]
    d[8] = I_du[u+1,v-1]
    d[9] = I_dv[u+1,v-1]
    d[10]= I_du[u+1,v+1]
    d[11]= I_dv[u+1,v+1]
    d[12]= I_du[u+3,v-1]
    d[13]= I_dv[u+3,v-1]
    d[14]= I_du[u+3,v+1]
    d[15]= I_dv[u+3,v+1]
    d[16]= I_du[u-5,v-3]
    d[17]= I_dv[u-5,v-3]
    d[18]= I_du[u-5,v+3]
    d[19]= I_dv[u-5,v+3]
    d[20]= I_du[u+5,v-3]
    d[21]= I_dv[u+5,v-3]
    d[22]= I_du[u+5,v+3]
    d[23]= I_dv[u+5,v+3]
    d[24]= I_du[u-1,v-5]
    d[25]= I_dv[u-1,v-5]
    d[26]= I_du[u-1,v+5]
    d[27]= I_dv[u-1,v+5]
    d[28]= I_du[u+1,v-5]
    d[29]= I_dv[u+1,v-5]
    d[30]= I_du[u+1,v+5]
    d[31]= I_dv[u+1,v+5]


def computeDescriptors(I, dims, maxima):
    begin = time.time()
    I_du = cv2.Sobel(I,cv2.CV_16S,1,0,ksize = 5, scale = 1, delta = 0,borderType = cv2.BORDER_DEFAULT)
    I_dv = cv2.Sobel(I,cv2.CV_16S,0,1,ksize = 5, scale = 1, delta = 0,borderType = cv2.BORDER_DEFAULT)
    end = time.time()
    print 'sobel filtering =', end-begin

    begin = time.time()
    for it in maxima:
        u = it.u
        v = it.v
        computeDescriptor(I_du,I_dv, u,v,it)
    end = time.time()
    print 'descriptor computing=', end-begin


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
    begin = time.time()
    blob_img = cv2.filter2D(img, -1, kernel2)
    corn_img = cv2.filter2D(img, -1, kernel)
    end = time.time()
    print 'blob&corner filtering=', end-begin

    dims = img.shape[0:2]  # dims[0] - width, dims[1] = height
   # extract maxima via non-maximum suppression

    I_temp = np.zeros((dims[0], dims[1]), dtype=np.int32)
    nonMaximumSuppression(blob_img,I_temp, dims, True, 1, maxima)
    nonMaximumSuppression(blob_img,I_temp, dims, False, 2, maxima)
    nonMaximumSuppression(corn_img,I_temp, dims, True, 3, maxima)
    nonMaximumSuppression(corn_img,I_temp, dims, False, 4,maxima)
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
def match_features(maxima1, i1, maxima2, i2,  now_prev):
    M = 150

    min_sad1 = 1000000
    min_sad2 = 1000000
    # begin = time.time()
    for i in xrange(len(maxima2)):  #try to make a vector of np, make a vector with u,v,c (probably)
        # end = time.time()
        if maxima1[i1].c == maxima2[i].c and abs(maxima2[i].u - maxima1[i1].u) < M and abs(maxima2[i].v - maxima1[i1].v) < M and (now_prev or abs(maxima2[i].u - maxima1[i1].u)<= 1):
            # print '%s exeption' %(end-begin)
            # visited[i] = True
            # end = time.time()
            sad = SAD_NP(maxima1[i1].d1, maxima2[i].d1)
            # print '%s sad' %(end - begin)
            if sad < min_sad1:
                min_sad1 = sad
                i2 = i

            elif sad < min_sad2:
                min_sad2 = sad

    if min_sad1 < (min_sad2*0.90):
        return True, i2
    else:
        return False, None

def SAD_NP(descr1, descr2):
   return np.sum(np.abs(descr1 - descr2))

def match_features_reorg(data_1,u1,v1, data_2, u2,v2, now_prev, M = 250):

    min_sad1 = 1000000
    min_sad2 = 1000000
    # make window M*M
    dat = data_1[u1,v1]
    x, y = data_2.shape
    margin = 5
    if now_prev:
         window = data_2[max(margin, u1 - int(ceil(M/2))):min(int(ceil(u1+M/2)), x),max(margin, v1-int(ceil(M/2))):min(v1+int(ceil(M/2)), y)]
    else:

         window = data_2[u1-10:u1+10,max(margin, v1-int(ceil(M/2))):min(v1+int(ceil(M/2)), y)]


    nz = window['c'].nonzero()
    non_zero_indices = np.transpose(nz)

    for index in non_zero_indices:
#    for u in range(window.shape[0]):
#         for v in range(window.shape[1]):
            u,v = index
            if dat['c'] == window[u,v]['c']: #and any(window[u,v]['descr'] != 0 ): # and (now_prev or abs(maxima2[i].u - maxima1[i1].u)<= 1):
                sad = SAD_NP(dat['descr'], window[u,v]['descr'])

                # check SAD of each descryptor
                if sad < min_sad1:
                    min_sad1 = sad
                    if now_prev:
                        u2, v2 = np.array((u,v)) + np.array((max(margin, u1 - ceil(M/2)), max(margin, v1 - ceil(M/2))))
                    else:
                        u2,v2 = np.array((u,v)) + np.array(((u1 -10), max(margin, v1 - ceil(M/2))))
                elif sad < min_sad2:
                   min_sad2 = sad

    if min_sad1 < (min_sad2*0.90):
        return True, u2,v2
    else:
         return False, None,None
    # return True, u2,v2


def match_two_pictures(maximaLeftp,maximaRightp):
    start = time.time()
    p =[]
    i = 0
    # files = 'matchingMovedImages_w-100.dat'
    for iLp in range (len(maximaLeftp)):
            iRp, iLc, iRc, iLp_new = [0, 0, 0, 0]
            tmp, iRp =  match_features(maximaLeftp, iLp, maximaRightp, iRp,  True)
            if not tmp:
                continue

            tmp, iLp_new = match_features(maximaRightp, iRp, maximaLeftp, iLp_new, True)
            if not tmp:
                continue
            if iLp_new == iLp:
                 p.append(p_match(maximaLeftp[iLp].u,maximaLeftp[iLp].v, maximaRightp[iRp].u,
                                       maximaRightp[iRp].v, maximaLeftp[iLp].c, maximaRightp[iRp].c))
                 # f = open(files, 'a+b')
                 # pickle.dump(p[i], f)
                 # f.close()
                 # i +=1
    print i
    end = time.time()

    print end-start
    return p
    M = 150

def match_two_pictures_reorg(data_left, data_right, maximaLeft, files = False):
    p = []
    if files:
        file = 'shelve_5cm.txt'
        f = open(file, 'w')
        j = 0
#random cycle
    mlen = len(maximaLeft)
    step = 100
    distance_lp_rp = np.zeros(step)


    start = time.clock()
    for i in range(0, mlen, int(mlen/step)):
        uLp,vLp = [maximaLeft[i].u,maximaLeft[i].v]
        uLp_new ,vLp_new = [0,0]

        uRp,vRp = [0,0]

        tmp, uRp,vRp =  match_features_reorg(data_left, uLp,vLp, data_right, uRp,vRp, False)
        if not tmp:
            continue

        tmp,uLp_new,vLp_new =  match_features_reorg(data_right, uRp,vRp, data_left, uLp_new,vLp_new, False)
        if not tmp:
            continue

        if uLp_new == uLp and vLp_new == vLp:

            cLp = data_left[uLp,vLp]['c']
            cRp = data_right[uRp,vRp]['c']
            distance_lp_rp[j] = sqrt((uLp - uRp)**2 + (vLp - vRp)**2)
            p.append(p_match( uLp,vLp,uRp,vRp, cLp,cRp ))
            data_left[uLp, vLp]['c'] = 0
            data_right[uRp, vRp]['c']= 0
            if files:
                f.write(str(uLp)+ ','+ str(vLp) + ',' +str(uRp)+ ',' +str(vRp) +'\n')
                j+=1


        l_r = int(np.max(distance_lp_rp))+4

    end = time.clock()
    print 'lp to rp', l_r

    print 'time for 150x150 window',end-start
    print 'len p= ', len(p)
#count distance
    #normal cycle with M = distance.
    # for iLp in range(1,len(data_left[data_left!=0])):
    i = 0
    start = time.clock()
    for maxima in maximaLeft:
        uLp,vLp = [maxima.u,maxima.v]
        uLp_new ,vLp_new = [0,0]
        uRp,vRp = [0,0]

        tmp, uRp, vRp = match_features_reorg(data_left, uLp,vLp, data_right, uRp,vRp, False, M = l_r )
        if not tmp:
            continue

        tmp, uLp_new, vLp_new = match_features_reorg(data_right, uRp,vRp, data_left, uLp_new,vLp_new, False, M = l_r)


        if uLp_new == uLp and vLp_new == vLp:
            cLp = data_left[uLp,vLp]['c']
            cRp = data_right[uRp,vRp]['c']
            p.append(p_match( uLp,vLp,uRp,vRp, cLp,cRp))
            if files:
                f.write(str(uLp)+ ','+ str(vLp) + ',' +str(uRp)+ ',' +str(vRp) +'\n')

            # f = open(files, 'a+b')
            # pickle.dump(p[j], f)
            # f.close()
                j+=1
    end = time.clock()
    if files:
        f.close()
    print 'time for the rest points', end-start
    return p


def match_four_pictures_reorg(data_left, data_right,data_left_cur,data_right_cur, maximaLeft, first_imgs = True, files = False):
    p = []
    if files:
        file = 'matchingImages_reorg_1.dat'
    if first_imgs:
    #random cycle
        mlen = len(maximaLeft)
        step = 100
        distance_lp_rp = np.zeros(step)
        distance_rp_rc = np.zeros(step)
        distance_rc_lc = np.zeros(step)
        distance_lc_lp = np.zeros(step)
        j = 0
        start = time.clock()
        for i in range(0, mlen, int(mlen/step)):
            uLp,vLp = [maximaLeft[i].u,maximaLeft[i].v]
            uLp_new ,vLp_new = [0,0]

            uRp,vRp = [0,0]
            uRc,vRc = [0,0]
            uLc,vLc = [0,0]

            tmp, uRp,vRp =  match_features_reorg(data_left, uLp,vLp, data_right, uRp,vRp, False)
            if not tmp:
                continue

            tmp,uRc,vRc =  match_features_reorg(data_right, uRp,vRp, data_right_cur, uRc, vRc, True)
            if not tmp:
                continue

            tmp, uLc,vLc = match_features_reorg(data_right_cur, uRc,vRc, data_left_cur, uLc,vLc, False)
            if not tmp:
                continue
            tmp, uLp_new, vLp_new = match_features_reorg(data_left_cur, uLc,vLc, data_left, uLp_new,vLp_new, True)
            if not tmp:
                continue

            if uLp_new == uLp and vLp_new == vLp:

                cLp = data_left[uLp,vLp]['c']
                cRp = data_right[uRp,vRp]['c']
                cLc = data_left_cur[uLc,vLc]['c']
                cRc = data_right_cur[uRc,vRc]['c']
                distance_lp_rp[j] = sqrt((uLp - uRp)**2 + (vLp - vRp)**2)
                distance_rp_rc[j] = sqrt((uRp - uRc)**2 + (vRp - vRc)**2)
                distance_rc_lc[j] = sqrt((uRc - uLc)**2 + (vRc - vLc)**2)
                distance_lc_lp[j] = sqrt((uLc - uLp)**2 + (vLc - vLp)**2)
                p.append(p_match_four( uLp,vLp,uRp,vRp, uLc, vLc, uRc, vRc, cLp,cRp, cLc,cRc))
                data_left[uLp, vLp]['c'] = 0
                data_right[uRp, vRp]['c']= 0
                data_left_cur[uLc, vLc]['c']=0
                data_right_cur[uRc,vRc]['c']=0
                if files:
                 f = open(file, 'a+b')
                 pickle.dump(p[j], f)
                 f.close()

                j+=1


        l_r = max(int(np.max(distance_lp_rp)), int(np.max(distance_rc_lc)))+4
        flow = max(int(np.max(distance_rp_rc)),int(np.max(distance_lc_lp)))+4
        lp_rp = np.max(distance_lp_rp) +4
        rp_rc = np.max(distance_rc_lc) +4
        rc_lc = np.max(distance_rp_rc) +4
        lc_lp = np.max(distance_lc_lp) +4
        end = time.clock()
        print 'lp to rp', l_r
        print 'rp to rc', flow



        print 'time for 150x150 window',end-start
        print 'len p= ', len(p)
    #count distance
    #normal cycle with M = distance.
    # for iLp in range(1,len(data_left[data_left!=0])):
    i = 0
    start = time.clock()
    for maxima in maximaLeft:
        uLp,vLp = [maxima.u,maxima.v]
        uLp_new ,vLp_new = [0,0]
        uRp,vRp = [0,0]
        uRc,vRc = [0,0]
        uLc,vLc = [0,0]

        tmp, uRp,vRp =  match_features_reorg(data_left, uLp,vLp, data_right, uRp,vRp, False, M = lp_rp )
        if not tmp:
            continue

        tmp,uRc,vRc =  match_features_reorg(data_right, uRp,vRp, data_right_cur, uRc, vRc, True, M = rp_rc)
        if not tmp:
            continue

        tmp, uLc,vLc = match_features_reorg(data_right_cur, uRc,vRc, data_left_cur, uLc,vLc, False, M = rc_lc)
        if not tmp:
            continue

        tmp, uLp_new, vLp_new = match_features_reorg(data_left_cur, uLc,vLc, data_left, uLp_new,vLp_new, True, M = lc_lp)
        if not tmp:
            continue

        if uLp_new == uLp and vLp_new == vLp:

            cLp = data_left[uLp,vLp]['c']
            cRp = data_right[uRp,vRp]['c']
            cLc = data_left_cur[uLc,vLc]['c']
            cRc = data_right_cur[uRc,vRc]['c']

            p.append(p_match_four( uLp,vLp,uRp,vRp, uLc, vLc, uRc, vRc, cLp,cRp, cLc,cRc))
            f = open(files, 'a+b')
            pickle.dump(p[j], f)
            f.close()
            j+=1
    end = time.clock()
    print 'time for the rest points', end-start
    return p


def match_four_pictures(maximaLeft, maximaRight, maximaLeftCur, maximaRightCur): #l, r l r
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
    # files = 'tryingOCV_4pict_p3.dat'
    for iLp in range (len(maximaLeft)):
        iRp, iLc, iRc, iLp_new = [0, 0, 0, 0]
        tmp, iRp =  match_features(maximaLeft, iLp, maximaRight, iRp, False)
        if not tmp:
            continue
        tmp,iRc =  match_features(maximaRight, iRp, maximaRightCur, iRc, True)
        if not tmp:
            continue
        tmp, iLc = match_features(maximaRightCur, iRc, maximaLeftCur, iLc, False)
        if not tmp:
            continue
        tmp, iLp_new = match_features(maximaLeftCur, iLc, maximaLeft, iLp_new, True)
        if not tmp:
            continue
        if iLp_new == iLp:
            uLp = maximaLeft[iLp].u
            uRp = maximaRight[iRp].u
            uLc = maximaLeftCur[iLc].u
            uRc = maximaRightCur[iRc].u
            vLp = maximaLeft[iLp].v
            vRp = maximaRight[iRp].v
            vLc = maximaLeftCur[iLc].v
            vRc = maximaRightCur[iRc].v

            cLp = maximaLeft[iLp].c
            cRp = maximaRight[iRp].c
            cLc = maximaLeftCur[iLc].c
            cRc = maximaRightCur[iRc].c

            p.append(p_match_four( uLp,vLp,uRp,vRp, uLc, vLc, uRc, vRc, cLp,cRp, cLc,cRc))
            # f = open(files, 'a+b')
            # pickle.dump(p[i], f)
            # f.close()
            # i+=1

    end = time.time()
    total = end-now
    print total
    return p

def reorganize_data(maxima, img):

    data = np.zeros(img.shape[0]* img.shape[1],dtype = [('c','i4'),('descr','object')] ).reshape(img.shape[0],img.shape[1])
    for i in range (len(data)):
        data[i] = (0, np.zeros(32,dtype = 'i4'))
    for i in range(len(maxima)):
        data[maxima[i].u,maxima[i].v] = (maxima[i].c, maxima[i].d1)

    return data



def tryAll():
    stringLeft= '/home/leyla/photo/IMG_8846.jpg'#'/home/leyla/pyProjects/вапвап/libviso/img/000001_left.jpg'
    stringRight = '/home/leyla/photo/IMG_8851.jpg'   #'/home/leyla/pyProjects/вапвап/libviso/img/000001_right.jpg'
    # stringRight = '/home/leyla/pyProjects/вапвап/libviso/img/000002_right.jpg'
    # stringLeft= '/home/leyla/pyProjects/вапвап/libviso/img/000002_left.jpg'
    stringRightCur = '/home/leyla/pyProjects/вапвап/libviso/img/000002_right.jpg'
    stringLeftCur= '/home/leyla/pyProjects/вапвап/libviso/img/000002_left.jpg'

    maximaRight =[]
    maximaLeft =[]

    maximaRightCur =[]
    maximaRightCur =[]
    maximaLeftCur =[]

    blob_imgRight=[]
    corn_imgRight=[]
    blob_imgLeft=[]
    corn_imgLeft=[]

    blob_imgRightCur=[]
    corn_imgRightCur=[]
    blob_imgLeftCur=[]
    corn_imgLeftCur=[]

    imgRight = cv2.imread(stringRight, 0)
    imgLeft = cv2.imread(stringLeft, 0)
    # imgRightCur = cv2.imread(stringRightCur, 0)
    # imgLeftCur = cv2.imread(stringLeftCur, 0)
    # print imgRight.shape
    # imgLeft = np.roll(imgRight,100, axis=1)
    # imgLeft = np.delete(imgLeft,np.s_[:100:],1)

    start = time.time()
    blob_imgRight, corn_imgRight =findFeatures(imgRight, maximaRight, blob_imgRight, corn_imgRight)  #right image
    blob_imgLeft, corn_imgLeft =findFeatures(imgLeft, maximaLeft, blob_imgLeft, corn_imgLeft)  #left image

    # blob_imgRightCur, corn_imgRightCur =findFeatures(imgRightCur, maximaRightCur, blob_imgRightCur, corn_imgRightCur)  #right image
    # blob_imgLeftCur, corn_imgLeftCur =findFeatures(imgLeftCur, maximaLeftCur, blob_imgLeftCur, corn_imgLeftCur)  #left image
    #
    end = time.time()
    print end - start

    keypointsRight =[]
    keypointsLeft =[]

    # keypointsRightCur =[]
    # keypointsLeftCur =[]

    for i in range (0,len(maximaRight)):
        keypointsRight.append(cv2.KeyPoint(y= maximaRight[i].u, x = maximaRight[i].v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))

    for i in range (0,len(maximaLeft)):
         # if maxima1[i].c == 0:
         keypointsLeft.append(cv2.KeyPoint(y = maximaLeft[i].u,x = maximaLeft[i].v,  _size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))

    # for i in range (0,len(maximaRightCur)):
    #     keypointsRightCur.append(cv2.KeyPoint(y= maximaRightCur[i].u, x = maximaRightCur[i].v,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
    #
    # for i in range (0,len(maximaLeftCur)):
    #      # if maxima1[i].c == 0:
    #      keypointsLeftCur.append(cv2.KeyPoint(y = maximaLeftCur[i].u,x = maximaLeftCur[i].v,  _size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))


    im_with_keypointsRight= cv2.drawKeypoints(imgRight, keypointsRight, np.array([]), (0,0,255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    im_with_keypointsLeft = cv2.drawKeypoints(imgLeft, keypointsLeft, np.array([]), (0,0,255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # im_with_keypointsRightCur= cv2.drawKeypoints(imgRightCur, keypointsRightCur, np.array([]), (0,0,255),
    #                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # im_with_keypointsLeftCur = cv2.drawKeypoints(imgLeftCur, keypointsLeftCur, np.array([]), (0,0,255),
    #                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    #p = match_two_pictures(maximaLeft, maximaRight) #l r

    #p = match_four_pictures(maximaLeft, maximaRight, maximaLeftCur, maximaRightCur)
    s = time.clock()
    dataLeft = reorganize_data(maximaLeft, imgLeft)
    dataRight = reorganize_data(maximaRight,imgRight)
    # dataLeftCur= reorganize_data(maximaLeftCur,imgLeftCur)
    # dataRightCur = reorganize_data(maximaRightCur,imgRightCur)

    e = time.clock()
    print e - s
    s = time.clock()

    p = match_two_pictures_reorg(dataLeft, dataRight, maximaLeft)
    #p = match_four_pictures_reorg(dataLeft,dataRight,dataLeftCur,dataRightCur,maximaLeft)
    # cv2.drawMatchesKnn expects list of lists as matches.
    e = time.clock()
    print e - s
    print len(p)
    # # p = []
    # #`
    # # files = 'matchingMovedImages.dat'
    # # f = open(files, 'rb')
    # # print len(p)
    # # #
    # # i = 0
    # # while True:
    # #     try:
    # #         p.append( pickle.load(f))
    # #     except (EOFError):
    # #         break
    # # f.close()
    # # print len(descr)
    keypoints_descr_Right=[]
    keypoints_descr_Left=[]

    keypoints_descr_RightCur=[]
    keypoints_descr_LeftCur=[]


    for i in range (0, len(p)):
     #    u1p,v1p,u2p,v2p,u1c,v1c,u2c,v2c = p[i]
         keypoints_descr_Right.append(cv2.KeyPoint(y = p[i].uRp, x = p[i].vRp,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
         keypoints_descr_Left.append(cv2.KeyPoint( y= p[i].uLp, x = p[i].vLp,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
         # keypoints_descr_RightCur.append(cv2.KeyPoint(y = p[i].uRc, x = p[i].vRc,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
         # keypoints_descr_LeftCur.append(cv2.KeyPoint( y= p[i].uLc, x = p[i].vLc,_size= 1, _angle = -1, _response=0, _octave=0, _class_id = -1))
         #

    kpts_descr_Right = cv2.drawKeypoints(imgRight, keypoints_descr_Right, np.array([]), (255,0,0),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpts_descr_Left = cv2.drawKeypoints(imgLeft, keypoints_descr_Left, np.array([]), (0,255,0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # kpts_descr_RightCur = cv2.drawKeypoints(imgRightCur, keypoints_descr_RightCur, np.array([]), (0,0,255),
    #                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # kpts_descr_LeftCur = cv2.drawKeypoints(imgLeftCur, keypoints_descr_LeftCur, np.array([]), (0,255,0),
    #                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # cv2.imshow('features matched in Right', kpts_descr_Right)
    # cv2.imshow('features matched in Left', kpts_descr_Left)


    #
    # cv2.imshow('features matched in RightC', kpts_descr_RightCur)
    # cv2.imshow('features matched in LeftC', kpts_descr_LeftCur)

    # cv2.imshow('kptsL', im_with_keypointsLeft)
    # cv2.imshow('kptsR', im_with_keypointsRight)
    # cv2.imshow('kptsLc', im_with_keypointsLeftCur)
    # cv2.imshow('kptsRc', im_with_keypointsRightCur)
    # #
    both0 = kpts_descr_Left.copy() #l r
    # both1 = np.hstack((kpts_descr_LeftCur,kpts_descr_RightCur))
    #
    # both2 = np.vstack((kpts_descr_Left,kpts_descr_LeftCur))
    # both3 = np.vstack((kpts_descr_Right,kpts_descr_RightCur))

    a = np.zeros(len(p))
    # files = 'tan.dat'
    # #
    #f = open(files, 'a+b')
    # print both0.shape
    for i in range (0, len(p),5):
        # cv2.line(both2,( int(p[i].vLp),int(p[i].uLp)),(int(p[i].vLc),int(p[i].uLc+196 )),(255,0,0),1)
        # cv2.line(both3,( int(p[i].vRp),int(p[i].uRp)),(int(p[i].vRc),int(p[i].uRc+196 )),(255,0,0),1)

        #cv2.line(both0,( int(p[i].vLp),int(p[i].uLp)),(int(p[i].vRp+imgLeft.shape[1]),int(p[i].uRp )),(255,0,0),1)
        cv2.line(both0,( int(p[i].vLp),int(p[i].uLp)),(int(p[i].vRp),int(p[i].uRp )),(255,0,0),1)
        if p[i].vLp-p[i].vRp != 0:
          a[i] = float(float((p[i].uLp - p[i].uRp)) / float((p[i].vLp-p[i].vRp)))
        else:
          a[i] = np.nan;

        # cv2.line(both1,( int(p[i].vLc),int(p[i].uLc)),(int(p[i].vRc+672),int(p[i].uRc )),(255,0,0),1)


    #     a[i] = (p[i].uLp - p[i].uRp) / (p[i].vLp-p[i].vRp)
    #     pickle.dump(a[i], f)
    # f.close()
    #  for i in range (0, len(p)):
    #      if abs(p[i].vLp-p[i].vRp) >100:
    #          pass
    #      # cv2.line(both0,( p[i].vLp,p[i].uLp),(p[i].vRp,p[i].uRp+192 ),(255,0,0),1)
    #      a[i] = float(float((p[i].uLp - p[i].uRp+196)) / float((p[i].vLp-p[i].vRp)))
    #      # pickle.dump(a[i], f)
    # f.close()
    # f = open(files, 'rb')
    # i = 0
    # while True:
    #      try:
    #          a.append( pickle.load(f))
    #      except (EOFError):
    #          break
    a = np.array(a)
    a[a==-np.inf] = 50
    a[np.isnan(a)] = 50
    a[a==np.inf] = 50
    # f.close()
    print a
    minimum = np.amin(a)
    maximum = np.amax(a)
    print 'min =', minimum, "max = ", maximum
    plt.hist(a,bins = 200)
    plt.title('plot_shelve_5cm')
    plt.xlabel("value")
    plt.ylabel('frequency')
    fig = plt.gcf()
    #
    plot_url = py.plot_mpl(fig, filename ='plot_shelve_5cm')
    #
    #

    # print len(a)
    #py.sign_in('username', 'api_key')
    # #
    # # print a
    # #     if p[i].c1 == 1:
    # #         cv2.line(both1,(p[i].v2p, p[i].u2p),(p[i].v1p, p[i].u1p+196),(0,255,0),1)
    # #     if p[i].c1 == 2:
    # #         cv2.line(both2,(p[i].v2p, p[i].u2p),(p[i].v1p, p[i].u1p+196),(0,100,100),1)
    # #     if p[i].c1 == 3:
    # #         cv2.line(both3,(p[i].v2p, p[i].u2p),(p[i].v1p, p[i].u1p+196),(255,200,0),1)


    plt.imshow(kpts_descr_Left, cmap = 'gray', interpolation = 'bicubic')

    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    plt.imshow(kpts_descr_Right,cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    # cv2.imshow("test1",both1)
    # cv2.imshow("test2",both2)
    # cv2.imshow("test3",both3)

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test',both0)
    k  = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('lines5cm.png',both0)
        cv2.destroyAllWindows()
    cv2.namedWindow('features matched in Right',cv2.WINDOW_NORMAL)
    cv2.imshow('features matched in Right', kpts_descr_Right)
    k  = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
          cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('right5cm.png',kpts_descr_Right)
        cv2.destroyAllWindows()
    cv2.namedWindow('features matched in Left',cv2.WINDOW_NORMAL)
    cv2.imshow('features matched in Left', kpts_descr_Left)
    k  = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('left5cm.png',kpts_descr_Left)
        cv2.destroyAllWindows()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    cv2.destroyAllWindows()

tryAll()