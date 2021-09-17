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

import pickle
import sys, getopt


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

        #flatten organized pointcloud into array width*height x 1
        pc_arr = pc_in.to_array()
        m,n = np.shape(pc_arr)

        # make the same array extended with a column of 1s for T matrix multiplication
        jedinice = np.ones((m,n+1))
        jedinice[:,0:3] = pc_arr

        # transform all points in rows of jedinice by multiplication with Transformation mat
        transformed = mat.dot(jedinice.T).T

        # remove column of ones
        transformed = transformed[:,0:3]

        window = 20
        width = 640
        height = 480
        indices = []
        for x in range(max(0,mid_x-window), min(mid_x+window, width)):
            for y in range(max(0,mid_y-window), min(mid_y+window, height)):
                indices.append(y * width + x)

        # tmps = transformed[indices]
        tmp = [transformed[i,:] for i in indices if not np.isnan(sum(transformed[i]))]
        print(np.shape(tmp))

        if len(tmp) > 0:
            mid = np.median(tmp, 0)
            print("found mid. tmp size: ", len(tmp))
            self.mid = mid
        else:
            print("Error... this is tmp")
            print(tmp)
        # center around the global mid point

        try:
            transformed -= self.mid
            # compute distance to mid point (norm of centered points)
            dists = np.linalg.norm(transformed, axis=1)

            # choose those within radius (e.g. for flower r is cca 1cm)
            indices = np.array([i for i in range(len(dists)) if dists[i] < radius])
            
            # find indices of this point in the organised point cloud
            # this is inverse of: index = width * y + x
            # with width = 640, x in width, height = 480, y in height
            x = indices % 640
            y = [(i - ix)/640 for i,ix in zip(indices, x)]

            # pack determined indices of organised point cloud into a list
            indices = [[ix,iy] for ix, iy in zip(x, y)]


            pc_out = pcl.PointCloud()

            self.limits['min'] = np.min(tmp,0)
            self.limits['max'] = np.max(tmp,0)

            pc_out.from_list(tmp)

            return pc_out, indices
        except ValueError:
            print("Error in detection")
            return False

    def filter_by_center_radius(self, pc_in, mat, radius):

        #flatten organized pointcloud into array width*height x 1
        pc_arr = pc_in.to_array()
        m,n = np.shape(pc_arr)

        # make the same array extended with a column of 1s for T matrix multiplication
        jedinice = np.ones((m,n+1))
        jedinice[:,0:3] = pc_arr

        # transform all points in rows of jedinice by multiplication with Transformation mat
        transformed = mat.dot(jedinice.T).T

        # remove column of ones
        transformed = transformed[:,0:3]

        # center around the global mid point
        transformed -= self.mid

        # compute distance to mid point (norm of centered points)
        dists = np.linalg.norm(transformed, axis=1)

        # choose those within radius (e.g. for flower r is cca 1cm)
        indices = np.array([i for i in range(len(dists)) if dists[i] < radius])
        
        # find indices of this point in the organised point cloud
        # this is inverse of: index = width * y + x
        # with width = 640, x in width, height = 480, y in height
        x = indices % 640
        y = [(i - ix)/640 for i,ix in zip(indices, x)]

        # pack determined indices of organised point cloud into a list
        indices = [[ix,iy] for ix, iy in zip(x, y)]
        
        return indices

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

    return trans


def label_image(tfpath, impath, pcdpath, fileid, imout, mid):

    trans = load_transform(tfpath, fileid)

    img = cv2.imread(impath)
    
    print("loading pc...")
    pc = pcl.load(pcdpath)

    mat = trans_to_mat.msg_to_se3(trans)

    r = Rectangle()
    r.mid = mid

    print("filtering pc...")
    indices = r.filter_by_center_radius(pc, mat, 0.012)

    print("indices")
    try:
        print(np.min(indices, 0))
        print(np.max(indices, 0))
        [x1,y1] = np.min(indices, 0)
        [x2,y2] = np.max(indices, 0)
        # plt.show()
    except ValueError:
        return False, [-1,-1,-1,-1]


    delta = int((r.width - r.height) / 2)
    crop_img = img[:, delta: r.width - delta]
    cv2.imwrite(imout, crop_img)

    x1 = max(0, x1-delta)
    x2 = min(r.height, x2-delta)
    
    fig, r.imhandle = plt.subplots()
    r.imhandle.imshow(crop_img)

    rect = patches.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=2, edgecolor='r', facecolor='none')
    
    r.imhandle.add_patch(rect)


    return(True, [x1,x2,y1,y2])

def main(folder_path, center_file, outfile):

    if not folder_path[-1] == "/":
        folder_path+="/"

    prefix = folder_path.split('/')[-2]

    impath = folder_path + "imgs_in/*.jpg"
    imglist = sorted(glob.glob(impath))
    n_images = len(imglist)
    pcds = folder_path + "pcds/*.pcd"
    pcdlist = sorted(glob.glob(pcds))

    impath_out = folder_path + "imgs_out/*.jpg"
    imglist_out = sorted(glob.glob(impath_out))

    limits = []
    impaths = []

    tfpath = folder_path+"flower_tfs.txt"
    
    # mid = [0.08702617, 0.50495207, 0.55962941] #flower3
    # mid = [0.07127107, 0.53958937, 0.54814767] #flower4
    # mid = [0.08643315, 0.51919327, 0.55886925] #flower5
    # mid = [0.04721904, 0.4836372,  0.57079311]  #flower6
    # mid = [0.03730104, 0.50511385, 0.57007093]  #flower7

    with open(center_file, "rb") as f:
        mid = pickle.load(f)

    count = 0
    # for fileid in range(5):
    for fileid in range(n_images):

        print("loading img... ", fileid)

        impath = imglist[fileid]
        imout = imglist_out[fileid]
        pcdpath = pcdlist[fileid]
        success, l = label_image(tfpath, impath, pcdpath, fileid, imout, mid)
        
        if success:
            count += 1
            limits.append(l)

            imname = impath.split('/')[-1]

            impaths.append(imname)

    print(limits)

    print("successful labels: ", count)

    generate_header = False
    with open(outfile, 'a+', newline='') as csvfile:
        reader = csv.reader(csvfile)
        generate_header = len(list(reader)) == 0
        

    with open(outfile, 'a+', newline='') as csvfile:
        fieldnames = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if generate_header:
            writer.writeheader()

        w = 480
        h = 480
        c = 'flower'

        for (ls, im) in zip(limits, impaths):
            writer.writerow({   'filename': prefix + '/imgs_out/' + im, 
                                'width': int(w),
                                'height': int(h),
                                'class': c,
                                'xmin': int(ls[0]),
                                'xmax': int(ls[1]),
                                'ymin': int(ls[2]),
                                'ymax': int(ls[3])
                                })



    plt.show()



if __name__ == '__main__':

    argv = sys.argv[1:]

    inpath = ''
    midfile = ''
    outfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:m:o:",["ifile=","midfile=","outfile="])
    except getopt.GetoptError:
        print( 'mark_object.py -i <inpath> -m <midfile> -o <outfile> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('mark_object.py -i <inpath> -m <midfile> -o <outfile>  ')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inpath = arg
        elif opt in ("-m", "--midfile"):
            midfile = arg
        elif opt in ("-o", "--outfile"):
            outfile = arg

    print("read args:" )
    print(inpath)
    print(midfile)
    print(outfile)

    main(inpath, midfile, outfile)