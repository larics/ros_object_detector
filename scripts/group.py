#!/usr/bin/env python3

import rospy
# import ros_numpy
from geometry_msgs.msg import Point, PointStamped

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import

from math import sqrt, isnan
import numpy as np

import pickle 
import random

import matplotlib._color_data as mcd
import sklearn.cluster as sklcl
import sklearn.mixture as sklmx

import timeit

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class Counter(object):

    def __init__(self, inpath, method, params):
        
        self.pepper_list = []

        with open(inpath, "rb") as f:
            self.points = pickle.load(f)

        print("shape1")
        print(np.shape(self.points))

        det_per_volume = 100. #single side
        # det_per_volume = 200. # from both sides, V motion

        len_pnts = len(self.points)
        short_len = int(len_pnts/2.)
        # self.points = self.points[::2]
        # self.points = self.points[short_len:]
        modulus = max(1,int(len(self.points)/det_per_volume))
        # print("shape2")
        # print(np.shape(self.points))
        # print("modulus = len data / 100 = ", modulus)

        self.data = []
        for ps in self.points[::modulus]:
            point = ps.point
            voxel = [point.x, point.y, point.z]
            # voxel_temp = [i for i in voxel]
            # voxel_temp = [i*100. for i in voxel]
            voxel_temp = [int(i*100) for i in voxel]
            self.data.append(voxel_temp)
        n_init = len(self.data)
        i = len(self.data)

        # # print("original set size: ", i)
        # set_size = 2*n_init
        # set_size = 500

        # break_flag = False
        # if len(self.data) < set_size:
        #     for rand_i in range(500):
        #         for ps in self.points[rand_i%modulus+1::modulus]:
        #             voxel = [ps.point.x, ps.point.y, ps.point.z]
        #             if i < set_size:
        #                 i+=1
        #                 noise = np.random.normal(1,0.01,3)
        #                 voxel_temp = [v*n for v,n in zip(voxel, noise)]
        #                 # voxel_temp = [i for i in voxel]
        #                 voxel_temp = [int(i*100) for i in voxel_temp]
        #                 # voxel_temp = [i*100. for i in voxel_temp]
        #                 self.data.append(voxel_temp)
        #             else:
        #                 break_flag = True
        #                 break
        #         if break_flag:
        #             break

        # # print("shape3")
        # # print(np.shape(self.data))

    def remove_outliers(self):

        data = np.asmatrix(self.data)

        xmid = round(np.mean(data[:,0])/100.) * 100
        ymid = round((np.mean(data[:,1]) + 50)/100.) * 100 - 50
        
        xmin = xmid - 50
        xmax = xmid + 50

        ymin = ymid - 100
        ymax = ymid + 100
                
        zmin = 35
        zmax = 100

        limits = [xmin, xmax, ymin, ymax, zmin, zmax]

        data_temp = []
        for row in self.data:
            keep = True
            for i_dim in range(3):
                # print(row[i_dim])
                # print(limits[i_dim*2])
                if (row[i_dim] > limits[i_dim*2]) and (row[i_dim] < limits[i_dim*2+1]):
                    pass
                else:
                    keep = False
            if keep:
                data_temp.append(row)

        data_temp = np.array(data_temp)
        self.data = data_temp



        # self.data_orig = self.data
        # # clf = IsolationForest(random_state=0)
        # # clf.fit(data)
        # # indata = clf.predict(data)
        
        # clf = LocalOutlierFactor()
        # indata = clf.fit_predict(data)


        # indata = [True  if i==1 else False for i in indata]
        # # print(indata)


        # data_in = data[indata]
        # data_out = data[[not i for i in indata]]

        
        # self.data = data_in


        # mins = [0,0,0]
        # maxs = [0,0,0]
        # for idim in range(3):
        #     mins[idim] = min(self.data[:,idim])
        #     maxs[idim] = max(self.data[:,idim])

        # data_temp = []
        # for item in self.data_orig:
        #     keep = True
        #     for dim in range(3):
        #         if item[dim] < mins[dim] or item[dim] > maxs[dim] :
        #             keep = False
        #     if keep:
        #         data_temp.append(item)
        
        # data_temp = np.array(data_temp)

        # data_orig = []
        # for p in self.points:
        #     data_orig.append([int(100.*p.point.x), int(100*p.point.y), int(100*p.point.z)])
        # data_orig = np.array(data_orig)
        # print(np.shape(data_orig))

        # plt.figure(1001)

        # ax1 = plt.subplot(1,1,1, projection='3d')
        # # ax1.scatter(data[:,0], data[:,1], data[:,2], marker='x')
        # ax1.scatter(data_in[:,0], data_in[:,1], data_in[:,2], marker='.', label='LOF inliers')
        # ax1.scatter(data_orig[:,0], data_orig[:,1], data_orig[:,2],  label='original')
        # ax1.scatter(data_out[:,0], data_out[:,1], data_out[:,2], marker='.',  label='LOF outliers')
        # # ax1.scatter(data_temp[:,0], data_temp[:,1], data_temp[:,2], marker='x')

        # ax1.plot3D([int(mins[0]), int(maxs[0])], [int(mins[1]), int(mins[1])], [int(mins[2]), int(mins[2])], 'gray', label='range inliers')

        # ax1.legend()
        # # ax1.legend(['LOF inliers','original',  'LOF outliers', 'range inliers'])


        # ax1.plot3D([int(mins[0]), int(maxs[0])], [int(maxs[1]), int(maxs[1])], [int(mins[2]), int(mins[2])], 'gray')
        # ax1.plot3D([int(mins[0]), int(maxs[0])], [int(mins[1]), int(mins[1])], [int(maxs[2]), int(maxs[2])], 'gray')
        # ax1.plot3D([int(mins[0]), int(maxs[0])], [int(maxs[1]), int(maxs[1])], [int(maxs[2]), int(maxs[2])], 'gray')

        # ax1.plot3D([int(mins[0]), int(mins[0])], [int(mins[1]), int(maxs[1])], [int(mins[2]), int(mins[2])], 'gray')
        # ax1.plot3D([int(maxs[0]), int(maxs[0])], [int(mins[1]), int(maxs[1])], [int(mins[2]), int(mins[2])], 'gray')
        # ax1.plot3D([int(mins[0]), int(mins[0])], [int(mins[1]), int(maxs[1])], [int(maxs[2]), int(maxs[2])], 'gray')
        # ax1.plot3D([int(maxs[0]), int(maxs[0])], [int(mins[1]), int(maxs[1])], [int(maxs[2]), int(maxs[2])], 'gray')

        # ax1.plot3D([int(mins[0]), int(mins[0])], [int(mins[1]), int(mins[1])], [int(mins[2]), int(maxs[2])], 'gray')
        # ax1.plot3D([int(maxs[0]), int(maxs[0])], [int(mins[1]), int(mins[1])], [int(mins[2]), int(maxs[2])], 'gray')
        # ax1.plot3D([int(mins[0]), int(mins[0])], [int(maxs[1]), int(maxs[1])], [int(mins[2]), int(maxs[2])], 'gray')
        # ax1.plot3D([int(maxs[0]), int(maxs[0])], [int(maxs[1]), int(maxs[1])], [int(mins[2]), int(maxs[2])], 'gray')

        # ax1.set_xlabel('x [cm]')
        # ax1.set_ylabel('y [cm]')
        # ax1.set_zlabel('z [cm]')
        # # ax1.legend(['LOF inliers',  'LOF outliers', 'range inliers'])
        # # plt.show()


        # print("indata by range shape:")
        # print(np.shape(data_temp))


        # self.data = data_temp



    def do_cluster(self):



    def plot_result(self, i_fig):




def pepper_per_plant(data, c):
    xmid = round(np.mean(data[:,0])/100.) * 100
    ymid = round((np.mean(data[:,1]) + 50)/100.) * 100 - 50

    xmin = xmid - 50
    xmax = xmid + 50

    ymin = ymid - 100
    ymax = ymid + 100
        
    xlim = [xmin, xmid, xmax]
    ylim = [ymin, ymin + 50, ymid, ymid + 50, ymax]

    for iy in range(len(ylim)-1):
        for ix in range(len(xlim)-1):
            print('-')
            ymin = ylim[iy]
            ymax = ylim[iy+1]
            xmin = xlim[ix]
            xmax = xlim[ix+1]
            for i in c:
                if (i[0,1]>ymin) and (i[0,1]< ymax):
                    if (i[0,0] > xmin) and (i[0,0] < xmax):
                        print (i)


if __name__ == '__main__':



    paths = [
            # "/home/mars/catkin_ws/src/ros_object_detector/ukrug_g_d.npz", 
            #  "/home/mars/catkin_ws/src/ros_object_detector/ukrug_v_less",
            #  "/home/mars/catkin_ws/src/ros_object_detector/ukrug_v_lesser",
            #  "/home/mars/catkin_ws/src/ros_object_detector/front_back",
            #  "/home/mars/catkin_ws/src/ros_object_detector/front_back_2",
            #  "/home/mars/catkin_ws/src/ros_object_detector/front_back_3",
            #  "/home/mars/catkin_ws/src/ros_object_detector/single_array_elipse_1",
            #  "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee",
            #  "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_2",

             "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_4_1",
             "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_4_2",
             "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_4_3",
             "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_4_4",
             "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_4_5",
             "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_4_6",
             "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_4_7",
             "/home/mars/catkin_ws/src/ros_object_detector/mrs_traj_eevee_4_8",
             
                ]
    corr = [
        # 36, 
        # 36, 
        # 29, 
        # 25,
        # 25,
        # 34,
        # 17,
        # 16,
        # 16,

        9,
        10,
        10,
        8,
        10,
        10,
        9,
        9,
        ]

    try:
        for cor, inpath in zip(corr, paths):
            for i_fig, (method, param) in enumerate(zip(methods, params)):
                try:
                    plt.close()
                except:
                    pass

                counter = Counter(inpath, method, param)
                counter.remove_outliers()
                counter.plot_voxel()
                counter.do_cluster()
                counter.plot_result(i_fig)
                plt.title(counter.methodname)
                
                c = counter.centres[:-1]
                n_peppers = len(c)
                print("estimate: ", n_peppers)
                print("actual: ", cor)

                data = np.asmatrix(counter.data)

                pepper_per_plant(data, c)

                print("\n\n")



                # for i in c:
                    # if i[0,1]>-10:
                        # print (i)
                # plt.figure(final)
                # ax1 = plt.subplot(1,1,1, projection='3d')
                # ax1.scatter(counter.centres[:,0], counter.centres[:,1], counter.centres[:,2])
                
            # plt.figure(final)
            # plt.legend(methods)
            
        # plt.show()
            # print(c)
            # print("\n")
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')


 

    # # xlim = [250, 300, 350]
    # xlim = [50, 100, 150]
    # # xlim = [-150, -100, -50]
    # # xlim = [-350, -300, -250]
    # # ylim = [150, 200, 250, 300, 350]
    # ylim = [-350, -300, -250, -200, -150]
    

    # print(c)