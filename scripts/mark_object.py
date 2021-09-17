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

import sys, getopt

import pickle

class Rectangle:
    def __init__(self):


        self.x1 = -1
        self.y1 = -1
        self.x2 = -1
        self.y2 = -1
        self.imhandle = None

        self.limits = {}

    def onclick(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata

    def offclick(self, event):
        self.x2 = event.xdata
        self.y2 = event.ydata
        
        print(self.x1)
        print(self.x2)
        print(self.y1)
        print(self.y2)
        
        x1 = min(self.x1, self.x2)
        x2 = max(self.x1, self.x2)

        y1 = min(self.y1, self.y2)
        y2 = max(self.y1, self.y2)

        rect = patches.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=2, edgecolor='b', facecolor='none')
        self.imhandle.add_patch(rect)
        print("drawn")
        plt.close()


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



    def filter_faster(self, pc_in, mat, radius):

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

        tmp = [transformed[i,:] for i in indices if not np.isnan(sum(transformed[i]))]

        if len(tmp) > 0:
            mid = np.median(tmp, 0)
            print("found mid. tmp size: ", len(tmp))
            self.mid = mid
        else:
            print("Error... this is tmp")
            print(tmp)

        try:
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


            pc_out = pcl.PointCloud()

            self.limits['min'] = np.min(tmp,0)
            self.limits['max'] = np.max(tmp,0)

            pc_out.from_list(tmp)

            return pc_out, indices
        except ValueError:
            print("Error in detection")
            return False


    def filter_and_transform(self, pc_in, mat):

        tmp = []

        x1 = int(self.x1)
        x2 = int(self.x2)
        y1 = int(self.y1)
        y2 = int(self.y2)

        for x in range(x1, x2):
            for y in range(y1, y2):
                ptemp = list(pc_in.get_point(x,y))
                ptemp.append(1.)
                p2 = (np.array(ptemp))
                p_trans = np.dot(mat, p2)

                if not np.isnan(sum(ptemp)):
                    tmp.append(p_trans[0:3])
                else:
                    print("nan")
        
        pc_out = pcl.PointCloud()
        # print(np.median(tmp,0))
        centred = tmp - np.mean(tmp, 0)

        # print(np.median(tmp,0))

        print("limits:")
        print(np.min(tmp,0))
        print(np.max(tmp,0))

        self.limits['min'] = np.min(tmp,0)
        self.limits['max'] = np.max(tmp,0)


        pc_out.from_list(tmp)

        return pc_out




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


def main(folder_path, outfile, fileid):

    if not folder_path[-1] == "/":
        folder_path+="/"
    
    prepath = folder_path+"flower_tfs.txt"
    trans = load_transform(prepath, fileid)

    mat = trans_to_mat.msg_to_se3(trans)
    impath = folder_path + "imgs_in/*.jpg"
    imglist = sorted(glob.glob(impath))
    img = cv2.imread(imglist[fileid])
    
    pcds = folder_path + "pcds/*.pcd"
    pcdlist = sorted(glob.glob(pcds))
    pcdpath = pcdlist[fileid]
    pc = pcl.load(pcdpath)
    
    r = Rectangle()

    fig, r.imhandle = plt.subplots()
    r.imhandle.imshow(img)

    cid1 = fig.canvas.mpl_connect('button_press_event', r.onclick)
    cid2 = fig.canvas.mpl_connect('button_release_event', r.offclick)

    plt.show()

    # labeled_object = r.filter_pointcloud(pc)
    # labeled_object = r.filter_and_transform(pc, mat)
    # labeled_object, indices = r.filter_by_radius(pc, mat, 0.01)
    labeled_object, indices = r.filter_faster(pc, mat, 0.01)

    print("mid")
    print(r.mid)

    with open(outfile, "wb") as f:
        pickle.dump(r.mid, f)

    fig, r.imhandle = plt.subplots()
    r.imhandle.imshow(img)

    x1 = min(r.x1, r.x2)
    x2 = max(r.x1, r.x2)
    y1 = min(r.y1, r.y2)
    y2 = max(r.y1, r.y2)
    rect = patches.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=2, edgecolor='b', facecolor='none')
    r.imhandle.add_patch(rect)


    print("indices")
    print(np.min(indices, 0))
    print(np.max(indices, 0))
    [x1,y1] = np.min(indices, 0)
    [x2,y2] = np.max(indices, 0)
    rect = patches.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=2, edgecolor='r', facecolor='none')
    
    circ = patches.Circle(r.mid, radius=1, facecolor='r')
    r.imhandle.add_patch(circ)

    for ind in indices:
        circ = patches.Circle(ind, radius=1, facecolor='g')
        r.imhandle.add_patch(circ)

    r.imhandle.add_patch(rect)
    plt.show()


    # visual = pcl.pcl_visualization.CloudViewing()
    # visual.ShowMonochromeCloud(labeled_object)#, b'cloud')
    # visual.ShowMonochromeCloud(pc)#, b'cloud')

    # v = True
    # while v:
    #     v = not(visual.WasStopped())


if __name__ == '__main__':

    argv = sys.argv[1:]

    inpath = ''
    outfile = ''
    fid = 50
    try:
        opts, args = getopt.getopt(argv,"hi:o:n:",["ifile=","ofile=","fileid="])
    except getopt.GetoptError:
        print( 'mark_object.py -i <inpath> -o <outfile>  [ -n <fileid> ]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('mark_object.py -i <inpath> -o <outfile> [ -n <fileid> ]')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inpath = arg
        elif opt in ("-o", "--ofile"):
            outfile = arg
        elif opt in ("-n", "--fileid"):
            fid = int(arg)

    print("read args:" )
    print(inpath)
    print(outfile)

    main(inpath, outfile, fid)