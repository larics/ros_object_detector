#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pcl
from pcl import pcl_visualization

from geometry_msgs.msg import TransformStamped
import tf2_ros

import trans_to_mat

import yaml

import csv

class Rectangle:
    def __init__(self):


        self.x1 = -1
        self.y1 = -1
        self.x2 = -1
        self.y2 = -1
        self.imhandle = None

        self.limits = {}
        
        self.width = 640
        self.height = 480




    def filter_pointcloud(self, pc_in):

        tmp = []

        x1 = int(self.x1)
        x2 = int(self.x2)
        y1 = int(self.y1)
        y2 = int(self.y2)

        for x in range(x1, x2):
            for y in range(y1, y2):
                ptemp = pc_in.get_point(x,y)
                if not np.isnan(sum(ptemp)):
                    tmp.append(ptemp)
        
        pc_out = pcl.PointCloud()
        # print(np.median(tmp,0))
        centred = tmp - np.mean(tmp, 0)

        # print(np.median(tmp,0))

        pc_out.from_list(tmp)

        return pc_out

    def filter_by_radius(self, pc_in, mat, radius):

        tmp = []

        x1 = int(self.x1)
        x2 = int(self.x2)
        y1 = int(self.y1)
        y2 = int(self.y2)

        mid_x = int((x2+x1)/2.)
        mid_y = int((y2+y1)/2.)

        print("mid_x: ", mid_x)
        print("mid_y: ", mid_y)

        self.mid = (mid_x, mid_y)

        window = 20
        for x in range(max(0,mid_x-window), min(mid_x+window, 640)):
            for y in range(max(0,mid_y-window), min(mid_y+window, 480)):
                ptemp = list(pc_in.get_point(x,y))
                ptemp.append(1.)
                p2 = (np.array(ptemp))
                p_trans = np.dot(mat, p2)
                if not np.isnan(sum(ptemp)):
                    tmp.append(p_trans[0:3])

        if len(tmp) > 0:
            mid = np.median(tmp, 0)
            print("found mid. tmp size: ", len(tmp))
            self.mid = mid
        else:
            print("Error... this is tmp")
            print(tmp)

        tmp = []
        indices = []
        # for x in range(x1, x2):
        #     for y in range(y1, y2):
        for x in range(0, 640):
            for y in range(0, 480):
                ptemp = list(pc_in.get_point(x,y))
                ptemp.append(1.)
                p2 = (np.array(ptemp))
                p_trans = np.dot(mat, p2)
                if np.linalg.norm(p_trans[0:3]-mid) < radius:
                    if not np.isnan(sum(ptemp)):
                        tmp.append(p_trans[0:3])
                        indices.append([x,y])

        pc_out = pcl.PointCloud()
        centred = tmp - np.mean(tmp, 0)

        print("limits")
        print(np.min(tmp,0))
        print(np.max(tmp,0))

        self.limits['min'] = np.min(tmp,0)
        self.limits['max'] = np.max(tmp,0)

        pc_out.from_list(tmp)

        return pc_out, indices


    def filter_by_center_radius(self, pc_in, mat, radius):

        mid = self.mid

        tmp = []
        indices = []
        for x in range(0, 640):
            for y in range(0, 480):
                ptemp = list(pc_in.get_point(x,y))
                ptemp.append(1.)
                p2 = (np.array(ptemp))
                p_trans = np.dot(mat, p2)
                if np.linalg.norm(p_trans[0:3]-mid) < radius:
                    if not np.isnan(sum(ptemp)):
                        tmp.append(p_trans[0:3])
                        indices.append([x,y])

        pc_out = pcl.PointCloud()
        centred = tmp - np.mean(tmp, 0)

        print("limits")
        print(np.min(tmp,0))
        print(np.max(tmp,0))

        self.limits['min'] = np.min(tmp,0)
        self.limits['max'] = np.max(tmp,0)

        pc_out.from_list(tmp)

        return pc_out, indices

def load_transform(fpath, fileid):

    tfs = []
    with open(fpath, 'r') as f: 
        docs = yaml.load_all(f) 
        for doc in docs:
            try: 
                tfs.append(doc['transform'])
            except TypeError:
                pass


    tf = tfs[fileid]

    trans = TransformStamped()                                              
    trans.transform.translation.x = tf['translation']['x']
    trans.transform.translation.y = tf['translation']['y']
    trans.transform.translation.z = tf['translation']['z']
    trans.transform.rotation.x = tf['rotation']['x']
    trans.transform.rotation.y = tf['rotation']['y']
    trans.transform.rotation.z = tf['rotation']['z']
    trans.transform.rotation.w = tf['rotation']['w']


    # trans = TransformStamped()                                              
    # trans.transform.translation.x = -0.02779859555256592                    
    # trans.transform.translation.y = 0.54458                                 
    # trans.transform.translation.z = 0.596741                                
    # trans.transform.rotation.x = -0.738788002226365                         
    # trans.transform.rotation.y = 0.17212062912091053                       
    # trans.transform.rotation.z = -0.15450400355698954                      
    # trans.transform.rotation.w = 0.6330049681339486      

    return trans


def label_image(tfpath, impath, pcdpath, fileid, imout):

    trans = load_transform(tfpath, fileid)

    img = cv2.imread(impath)
    
    print("loading pc...")
    pc = pcl.load(pcdpath)

    mat = trans_to_mat.msg_to_se3(trans)


    r = Rectangle()
    # fig, r.imhandle = plt.subplots()
    # r.imhandle.imshow(img)

    r.mid = [0.08229677, 0.51770914, 0.56806329]

    print("filtering pc...")
    labeled_object, indices = r.filter_by_center_radius(pc, mat, 0.015)


    fig, r.imhandle = plt.subplots()
    r.imhandle.imshow(img)

    print("indices")
    print(np.min(indices, 0))
    print(np.max(indices, 0))
    [x1,y1] = np.min(indices, 0)
    [x2,y2] = np.max(indices, 0)
    rect = patches.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=2, edgecolor='r', facecolor='none')
    
    # for ind in indices:
    #     circ = patches.Circle(ind, radius=1, facecolor='g', alpha=0.3)
    #     r.imhandle.add_patch(circ)

    r.imhandle.add_patch(rect)
    # plt.show()

    delta = (r.width - r.height) / 2
    crop_img = img[delta: r.width - delta, :]
    cv2.imwrite(imout, crop_img)

    x1 = max(0, x1-delta)
    x2 = min(r.height, x2-delta)
    
    
    return([x1,x2,y1,y2])

def main():

    folder_path = "/home/mars/data/flower_data/flowers2/"
    impath = folder_path + "imgs_in/*.jpg"
    imglist = sorted(glob.glob(impath))
    pcds = folder_path + "pcds/*.pcd"
    pcdlist = sorted(glob.glob(pcds))

    impath_out = folder_path + "imgs_out/*.jpg"
    imglist_out = sorted(glob.glob(impath))
    

    limits = []
    impaths = []

    for fileid in range(7):

        print("loading img... ", fileid)

        tfpath = folder_path+"flower_tfs"
        impath = imglist[fileid]
        imout = imglist_out[fileid]
        pcdpath = pcdlist[fileid]
        l = label_image(tfpath, impath, pcdpath, fileid, imout)
        limits.append(l)


        imname = impath.split('/')[-1]

        impaths.append(imname)

    print(limits)

    with open('labels_flowers2.csv', 'w', newline='') as csvfile:
        fieldnames = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        w = 480
        h = 480
        c = 'flower'

        for (ls, im) in zip(limits, impaths):
            writer.writerow({   'filename': im, 
                                'width': w,
                                'height': h,
                                'class': c,
                                'xmin': ls[0],
                                'xmax': ls[1],
                                'ymin': ls[2],
                                'ymax': ls[3]
                                })



    # plt.show()



if __name__ == '__main__':

    main()

    # fileid = 3
