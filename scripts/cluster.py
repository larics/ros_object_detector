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

def find_mean_member(cluster_members):

    cm = cluster_members
    # print(np.shape(cm[:,0,1]))
    # print(cm[:,0])
    return [np.mean(cm[:,0,0]), np.mean(cm[:,0,1]), np.mean(cm[:,0,2])]
    # return np.asarray([np.mean(cm[:,0,0]), np.mean(cm[:,0,1]), np.mean(cm[:,0,2])])


class Counter(object):

    def __init__(self, inpath, method, params):
        
        self.pepper_list = []
        self.methodname = None

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

        # print("original set size: ", i)
        set_size = 2*n_init
        set_size = 500

        break_flag = False
        if len(self.data) < set_size:
            for rand_i in range(500):
                for ps in self.points[rand_i%modulus+1::modulus]:
                    voxel = [ps.point.x, ps.point.y, ps.point.z]
                    if i < set_size:
                        i+=1
                        noise = np.random.normal(1,0.01,3)
                        voxel_temp = [v*n for v,n in zip(voxel, noise)]
                        # voxel_temp = [i for i in voxel]
                        voxel_temp = [int(i*100) for i in voxel_temp]
                        # voxel_temp = [i*100. for i in voxel_temp]
                        self.data.append(voxel_temp)
                    else:
                        break_flag = True
                        break
                if break_flag:
                    break

        # print("shape3")
        # print(np.shape(self.data))

        self.set_method(method, params, rand_i)

        # print(self.data)

    def set_method(self, method = None, params = None, point_augment = None):
        if not method is None:
            self.methodname = method
            randint_ = random.randint(0,100)

            if 'KMeans' in self.methodname:
                if not params is None:
                    n = params
                    self.method = sklcl.KMeans(n_clusters = n['n_clusters'],random_state=randint_)
                else:
                    self.method = sklcl.KMeans(random_state=0)

            elif 'DBSCAN' in self.methodname:
                if not params is None:
                    n = params
                    self.method = sklcl.DBSCAN(eps=n['eps'], min_samples=n['min_samples'],random_state=randint_)
                else:
                    self.method = sklcl.DBSCAN()

            elif 'OPTICS' in self.methodname:
                if not params is None:
                    n = params
                    if not point_augment is None:
                        min_samples = 2*(point_augment+2)
                        # min_samples = 3*(point_augment)
                        
                        self.method = sklcl.OPTICS(max_eps=n['max_eps'], min_samples=min_samples, metric='manhattan', min_cluster_size=min_samples)
                        
                        print("min samples: ", min_samples)
                    else:
                        self.method = sklcl.OPTICS(max_eps=n['max_eps'], min_samples=n['min_samples'], metric='manhattan', min_cluster_size=n['min_samples'])
                    # self.method = sklcl.OPTICS(max_eps=n['max_eps'], min_samples=n['min_samples'])
                    
                else:
                    self.method = sklcl.OPTICS()


            elif 'AffinityPropagation' in self.methodname:

                if not params is None:
                    n = params
                    self.method = sklcl.AffinityPropagation(damping=n['damping'],random_state=randint_)
                else:
                    self.method = sklcl.AffinityPropagation()


            elif 'MeanShift' in self.methodname:
                if not params is None:
                    n = params
                    self.method = sklcl.MeanShift(bandwidth=0.1,cluster_all=False)
                    # self.method = sklcl.MeanShift(bandwidth=n['bandwidth'])#,random_state=randint_)
                else:
                    self.method = sklcl.MeanShift()

            elif "SpectralClustering" in self.methodname:
                if not params is None:
                    n = params
                    self.method = sklcl.SpectralClustering(n_clusters=n['n_clusters'],random_state=randint_)
                else:
                    self.method = sklcl.SpectralClustering()

            elif "Birch" in self.methodname:
                if not params is None:
                    n = params
                    self.method = sklcl.Birch(threshold=n['threshold'], n_clusters=n['n_clusters'])
                else:
                    self.method = sklcl.Birch()

            elif "AgglomerativeClustering" in self.methodname:
                if not params is None:
                    n = params
                    self.method = sklcl.AgglomerativeClustering(n_clusters=n['n_clusters'], linkage=n['linkage'])
                    self.n_clusters = n['n_clusters']
                else:
                    self.method = sklcl.AgglomerativeClustering()
                    self.n_clusters = 2

            elif "GaussianMixture" in self.methodname:
                if not params is None:
                    n = params
                    self.method = sklmx.GaussianMixture(covariance_type=n['covariance_type'])#, n_components=n['n_components'],random_state=randint_)
                else:
                    self.method = sklmx.GaussianMixture()
            else:
                print("clustering with KMeans")
                self.method = sklcl.KMeans(random_state=0)
                self.methodname = 'KMeans'
        else:
            print("clustering with KMeans")
            self.method = sklcl.KMeans(random_state=0)
            self.methodname = 'KMeans'

        print(self.methodname)

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
        
        # if "Agglom" in self.methodname:
        #     # data = np.asmatrix(deepcopy(self.data))
        #     np.random.shuffle(self.data)

        data = np.asmatrix(self.data)
        # print("data")
        # print(np.shape(data))

        #voxelizacija...
        # data = float(int(data*100)/100.)

        start = timeit.timeit()
        self.method.fit(data)
        end = timeit.timeit()

        # print("duration: ", end-start)


        try:
            self.labels = self.method.labels_
        except:
            print("no labels in method", self.methodname)
            self.labels = self.method.predict(data)

        self.lset = set(self.labels)
        
        clustersizes = []
        for l in self.lset:
            clustersizes.append(len([i for i in self.labels if i == l]))

        for l,cs in zip(self.lset, clustersizes):
            if cs < 5:
                for i, lab in enumerate(self.labels):
                    if lab == l:
                        self.labels[i] = -1
        self.lset = set(self.labels)

        clustersizes = []
        for l in self.lset:
            clustersizes.append(len([i for i in self.labels if i == l]))


        try:
            self.centres = self.method.cluster_centers_
        except AttributeError:
            print("no centres in method", self.methodname)
            self.centres = []
            for label in self.lset:
                indices = [i for i, x in enumerate(self.labels) if x == label]
                cluster_members = np.asarray([data[i] for i in indices])
                center = find_mean_member(cluster_members)
                self.centres.append(center)
            self.centres = np.asmatrix(self.centres)
        print("no of peppers: ", len(self.centres)-1)
        
        # print("lset sizes: ", clustersizes)
        
        # end = timeit.timeit()
        # print("duration overall: ", end-start)
        # print("---------------------------------")
        # print("\n\n")



    def plot_result(self, i_fig):

        fig = plt.figure(i_fig)
        ax1 = plt.subplot(1,1,1, projection='3d')
        # ax2 = plt.subplot(2,1,2, projection='3d')

        color_keys = mcd.CSS4_COLORS.keys()
        diss = []

        color_keys = list(color_keys)

        n = len(self.lset)
        ncols = []
        color_keys_2 = []
        for i in range(n):
            ncols.append((i+3)*3)
        for ncol in ncols:
            color_keys_2.append(color_keys[ncol])

        new_centres = []
        i_iter = 0
        for label, center, color in zip(self.lset, self.data, color_keys_2):
            if i_iter < len(self.lset)-1:
                i_iter+=1
                indices = [i for i, x in enumerate(self.labels) if x == label]
                cluster_members = np.asarray([self.data[i] for i in indices])

                cn = mcd.CSS4_COLORS[color]
                cm = cluster_members

                ax1.scatter(cm[:,0], cm[:,1], cm[:,2], color=cn)
                ds = [0,0,0]
                for idim in range(3):
                    ds[idim] = max(cm[:,idim]) - min(cm[:,idim])

                diss.append(ds)

        ax1.legend([str(i) for i,l in enumerate(self.lset)])

        # try:
        i_iter = 0

        for center, color in zip(self.centres, color_keys_2):
            if i_iter < len(self.lset)-1:
                i_iter+=1
                cn = mcd.CSS4_COLORS[color]
                if len(np.shape(center)) == 1:
                    ax1.scatter(center[0], center[1], center[2], color=cn, s=100   )
                else: 
                    ax1.scatter(center[0,0], center[0,1], center[0,2], color=cn, s=100   )
        # except IndexError:
        #     pass;

        # print(diss)
        # plt.suptitle(self.title)
        # plt.show()

        ax1.set_xlabel('x [cm]')
        ax1.set_ylabel('y [cm]')
        ax1.set_zlabel('z [cm]')


    def plot_voxel(self):

        data = np.asmatrix(self.data)

        limits = []
        for dim in range(3):
            limits.append([min(data[:,dim]), max(data[:,dim])])

        # print (limits)

    def plot(self,data):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cols = []
        for pepper_msg in self.pepper_list[::2]:
            pepper = pepper_msg.point
            x,y,z = pepper.x, pepper.y, pepper.z
            # x,y,z = pepper.position.x, pepper.position.y, pepper.position.z
            d = 0.04
            # ax.plot([x-d, x+d], [y, y], [z, z])
            # c=plt.gca().lines[-1].get_color()
            # cols.append(c)
            # ax.plot([x, x], [y-d, y+d], [z, z], color=c)
            # ax.plot([x, x], [y, y], [z-d, z+d], color=c)
            # ax.scatter(x,y,z, color=c)
            ax.scatter(x,y,z)

        plt.show()
        # for i,h in enumerate(pepper.hist):
        #     ax.scatter(h[0],h[1],h[2], marker='.', color=c, linewidths=i)


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


    n_clusters = 15
    methods = [
                # 'AffinityPropagation',
                # # 'MeanShift',
                # # 'SpectralClustering', # SPOROOO
                # # 'KMeans', 
                # 'Birch',
                # # 'AgglomerativeClusteringWard',
                # # 'AgglomerativeClusteringComplete',
                # # 'AgglomerativeClusteringAverage',
                # 'GaussianMixtureFull',
                # # 'GaussianMixtureTied',
                # # 'GaussianMixtureDiag',
                # # 'GaussianMixtureSpher',
                'OPTICS',
                ]
    params = [  
                # {'damping':0.6},
                # # {'bandwidth':0.02}, 
                # # {'n_clusters':n_clusters},
                # # {'n_clusters':n_clusters},
                # {'threshold':1, 'n_clusters':None},
                # # {'n_clusters':n_clusters, 'linkage':'ward'},
                # # {'n_clusters':n_clusters, 'linkage':'complete'},
                # # {'n_clusters':n_clusters, 'linkage':'average'},
                # {'covariance_type':'full', 'n_components':n_clusters},
                # # {'covariance_type':'tied', 'n_components':n_clusters},
                # # {'covariance_type':'diag', 'n_components':n_clusters},
                # # {'covariance_type':'spherical', 'n_components':n_clusters},
                {'min_samples':2*3, 'max_eps':4.},
                ]



    # inpath = "/home/mars/catkin_ws/src/ros_object_detector/scripts/p05.npz"
    # inpath = "/home/mars/loc_40"
    # inpath = "/home/mars/catkin_ws/src/ros_object_detector/peppers_loc.npz"


    # inpath = "/home/mars/catkin_ws/src/ros_object_detector/ukrug" #36
    inpath = "/home/mars/catkin_ws/src/ros_object_detector/ukrug_g_d.npz" #36
    # inpath = "/home/mars/catkin_ws/src/ros_object_detector/ukrug_v_less" #29?
    # inpath = "/home/mars/catkin_ws/src/ros_object_detector/ukrug_v_lesser" #25?
    # inpath = "/home/mars/catkin_ws/src/ros_object_detector/front_back" #25?

    final = len(methods)
    # plt.figure(final)
    # ax1 = plt.subplot(1,1,1, projection='3d')

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
                # counter.outliers_by_range()
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