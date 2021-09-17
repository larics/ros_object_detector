#!/usr/bin/env python3

# from sklearn.cluster import KMeans
import sklearn.cluster as sklcl
import sklearn.mixture as sklmx
import numpy as np
import scipy.io

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib._color_data as mcd

from tf.transformations import quaternion_from_matrix, quaternion_matrix

import os, sys

import random
import time

from optimize_class import Fminsearch

from copy import deepcopy

def pos_q_2_pos_vec(row):
    s = []
    for i in [0,1,2]:
        s.append(row[i])
    
    x = row[3]
    y = row[4]
    z = row[5]
    w = row[6]
    V = [0, 0, 0]

    V[0] = 2 * (x * z - w * y)
    V[1] = 2 * (y * z + w * x)
    V[2] = 1 - 2 * (x * x + y * y)
    
    V = np.array(V)
    V /= 100*np.linalg.norm(V)
    for i in [0,1,2]:
        s.append(V[i])

    return s


def find_closest_member(cluster_members, center):
    min_dist = -1.
    min_index_subset = -1
    for i in xrange(len(cluster_members)):
        dist = np.linalg.norm(cluster_members[i, 0:3] - np.asarray(center))  
        if (min_dist == -1.) or dist < min_dist:
            min_dist = dist
            min_index_subset = i
    return min_index_subset

def find_mean_member(cluster_members):

    cm = cluster_members
    return np.asarray([np.mean(cm[:,0]), np.mean(cm[:,1]), np.mean(cm[:,2])])


def mean(a):
    return sum(a) / len(a)

class Clustering:

    def __init__(self, inpath, outpath, filename, *args, **kwargs):


        tmpdict = {}
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                tmpdict[key] = value
        else:
            print "clustering to 10 clusters"
            self.n_clusters = 10


        if len(kwargs) > 0:

            if "varname" in tmpdict.keys():
                varname = tmpdict["varname"]
            else:
                varname = "data"

            if 'method' in kwargs.keys():
                self.methodname = kwargs['method']

                randint_ = random.randint(0,100)
                if 'KMeans' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.KMeans(n_clusters = n['n_clusters'],random_state=randint_)
                    else:
                        self.method = sklcl.KMeans(random_state=0)

                if 'DBSCAN' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.DBSCAN(eps=n['eps'], min_samples=n['min_samples'],random_state=randint_)
                    else:
                        self.method = sklcl.DBSCAN()

                if 'OPTICS' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.OPTICS(max_eps=n['max_eps'], min_samples=n['min_samples'],random_state=randint_)
                    else:
                        self.method = sklcl.OPTICS()


                if 'AffinityPropagation' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.AffinityPropagation(damping=n['damping'],random_state=randint_)
                    else:
                        self.method = sklcl.AffinityPropagation()


                if 'MeanShift' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.MeanShift(bandwidth=n['bandwidth'],random_state=randint_)
                    else:
                        self.method = sklcl.MeanShift()

                if "SpectralClustering" in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.SpectralClustering(n_clusters=n['n_clusters'],random_state=randint_)
                    else:
                        self.method = sklcl.SpectralClustering()

                if "Birch" in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.Birch(threshold=n['threshold'], n_clusters=n['n_clusters'])
                    else:
                        self.method = sklcl.Birch()

                if "AgglomerativeClustering" in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.AgglomerativeClustering(n_clusters=n['n_clusters'], linkage=n['linkage'])
                        self.n_clusters = n['n_clusters']
                    else:
                        self.method = sklcl.AgglomerativeClustering()
                        self.n_clusters = 2

                if "GaussianMixture" in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklmx.GaussianMixture(covariance_type=n['covariance_type'], n_components=n['n_components'],random_state=randint_)
                    else:
                        self.method = sklmx.GaussianMixture()


                else:
                    pass
        else:
            print "clustering with KMeans"
            self.method = sklcl.KMeans(random_state=0)
            self.methodname = 'KMeans'

        print"------------"
        print(self.methodname)



        # if 'n_clusters' in tmpdict.keys():
        #     self.n_clusters =  tmpdict['n_clusters']
        # else:
        #     self.n_clusters = 10


        try:
            self.pos_quat = self.get_data_pos_quat(mat, varname)
            self.pos_quat_saved = self.pos_quat
        except KeyError as e:
            raise e
        self.pos_dir_vec = np.array([pos_q_2_pos_vec(row) for row in self.pos_quat])

        # self.method = sklcl.KMeans(n_clusters=self.n_clusters, random_state=randint_)

        self.filename = filename
        self.outpath = outpath
        title = self.filename.split('.mat')
        self.title = title[0]

        self.centres = []
        self.centres_q = []
        self.centres_dirvec = []


    def get_centres(self):
        return self.centres

    def set_centres_q(self, new_centres_q):
        self.centres_q = new_centres_q

    def get_data_pos_quat(self, mat, varname):

        pos_quat = None

        if "data" in varname:
            try:
                pos_quat = mat[varname]
            except KeyError:
                print "No variable named ", varname, " Please provide existing variable name in mat file"
                raise KeyError("No variable named "+varname+" Please provide existing variable name in mat file")
        elif "CalTx" in varname or "TP_master" in varname or "TP_slave" in varname:
            try:
                caltx = mat[varname]
            except KeyError:
                raise KeyError("No variable named "+varname+" Please provide existing variable name in mat file")
        elif "Kuka" in varname:
            try:
                temp = mat[varname]
                [k4,k4,m,n] = np.shape(temp)
                temp2 = np.zeros((4,4,m*n))
                for i in range(m):
                    for j in range(n):
                        temp2[:,:, (i-1) * n + j] = temp[:,:,i,j]
                caltx = temp2
            except KeyError:
                raise KeyError("No variable named "+varname+" Please provide existing variable name in mat file")
        else:
            raise KeyError("No variable named "+varname+" Please provide existing variable name in mat file")

        if pos_quat is None:
            m = max(np.shape(caltx))
            pos_quat = np.empty((m,7))
            
            a = np.empty((1, 7))

            caltx = np.transpose(caltx, (2,0,1))

            for i, mtrx in enumerate(caltx):
                q = quaternion_from_matrix(mtrx)
                p = mtrx[0:3,3].transpose()
                # print np.shape(a[0:3])
                # a[1,0:3] = np.resize(p,(1,3))
                a[0,0:3] = p
                a[0,3:] = q
                pos_quat[i,:] = a

        return pos_quat

    def do_cluster(self):
        
        if "Agglom" in self.methodname: 
            data = deepcopy(self.pos_quat_saved)
            np.random.shuffle(data)
            # np.random.shuffle(self.pos_quat)
            self.pos_quat = data[0:-2,:]
            self.pos_dir_vec = np.array([pos_q_2_pos_vec(row) for row in self.pos_quat])

        self.method.fit(self.pos_quat[:,0:3])

        try:
            self.labels = self.method.labels_
        except:
            print "no labels in method", self.methodname
            self.labels = self.method.predict(self.pos_quat[:,0:3])

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
        except:
            print "no centres in method", self.methodname
            for label in self.lset:
                indices = [i for i, x in enumerate(self.labels) if x == label]
                cluster_members = np.asarray([self.pos_dir_vec[i] for i in indices])
                center = find_mean_member(cluster_members)
                self.centres.append(center)
        print "no of centres: ", len(self.centres)
        print "no of labels: ", len(self.lset)
        print "lset sizes: ", clustersizes
        



    def add_missing_values_cluster_q(self):
        self.centres_dirvec = []
        self.centres_dirvec = [pos_q_2_pos_vec(row) for row in self.centres_q]
        self.centres_dirvec = np.asarray(self.centres_dirvec)


    def add_missing_values(self):
        #hard coded center = closest_point(center)
        self.centres_q = []
        self.centres_dirvec = []
        for label, center in zip(self.lset, self.centres):
        
            indices = [i for i, x in enumerate(self.labels) if x == label]
            cluster_members = np.asarray([self.pos_dir_vec[i] for i in indices])

            min_index_subset = find_closest_member(cluster_members, center)
            min_index_all = indices[min_index_subset]
            
            centre_q = self.pos_quat[min_index_all]
            centre_dirvec = cluster_members[min_index_subset]

            self.centres_q.append(centre_q)
            self.centres_dirvec.append(centre_dirvec)
        
        self.centres_q = np.asarray(self.centres_q)
        self.centres_dirvec = np.asarray(self.centres_dirvec)
    


    def plot_result(self, i_fig):

        if self.centres_q == []:
            self.add_missing_values()

        fig = plt.figure(i_fig)
        ax1 = plt.subplot(2,1,1, projection='3d')
        ax2 = plt.subplot(2,1,2, projection='3d')

        color_keys = mcd.CSS4_COLORS.keys()

        for label, center, color in zip(self.lset, self.centres_dirvec, color_keys):
        
            indices = [i for i, x in enumerate(self.labels) if x == label]
            cluster_members = np.asarray([self.pos_dir_vec[i] for i in indices])

            cn = mcd.CSS4_COLORS[color]

            cm = cluster_members
            ax1.quiver(cm[:,0], cm[:,1], cm[:,2], cm[:,3], cm[:,4], cm[:,5], color=cn)

            i = center    
            ax2.quiver(i[0], i[1], i[2], i[3], i[4], i[5], color=cn)

        plt.suptitle(self.title)


    def save_plot(self):
        plt.savefig(self.outpath+self.title)

    def save_to_mat(self):

        if self.centres_q == []:
            self.add_missing_values()

        self.centres_t = []
        for c in self.centres_q:
            t = np.asmatrix(quaternion_matrix(c[3:]))
            t[0:3, 3] = np.reshape(c[0:3], (3,1))
            self.centres_t.append(t)

        self.centres_t = np.array(self.centres_t)
        self.centres_t = self.centres_t.transpose((1,2,0))
        name = self.outpath + "centres_" + self.title + "_" + str(self.n_clusters) + ".mat"
        # scipy.io.savemat(name, {'c_c':self.centers})
        scipy.io.savemat(name, {'c_c':self.centres_t})



if __name__ == '__main__':
    
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        outpath = sys.argv[2]
    else:
        path = "/home/marsela/Documents/MATLAB/cluster/VU_exp_q/"
        outpath = "/home/marsela/Documents/MATLAB/cluster/VU_exp_q_centers/"

    all_files = os.listdir(path)
    caltx_files = [i for i in all_files if 'caltx' in i]

    krug_vrtnja = [i for i in caltx_files if "krug_vrtnj" in i and not "okoosi" in i]
    benchmark = np.asarray([ 0.01155917, -0.15891211,  0.00319799])


    n_clusters = 15
    methods = [ # 'AffinityPropagation',
                # 'MeanShift',
                # 'SpectralClustering', # SPOROOO
                
                
                # 'KMeans', 
                # 'Birch',
                'AgglomerativeClusteringWard',
                # 'AgglomerativeClusteringComplete',
                # 'AgglomerativeClusteringAverage',
                # 'GaussianMixtureFull',
                # 'GaussianMixtureTied',
                # 'GaussianMixtureDiag',
                # 'GaussianMixtureSpher',
                ]
    params = [  # {'damping':0.95},
                # {'bandwidth':0.02}, 
                # {'n_clusters':n_clusters},
                
                
                # {'n_clusters':n_clusters},
                # {'threshold':0.01, 'n_clusters':n_clusters},
                {'n_clusters':n_clusters, 'linkage':'ward'},
                # {'n_clusters':n_clusters, 'linkage':'complete'},
                # {'n_clusters':n_clusters, 'linkage':'average'},
                # {'covariance_type':'full', 'n_components':n_clusters},
                # {'covariance_type':'tied', 'n_components':n_clusters},
                # {'covariance_type':'diag', 'n_components':n_clusters},
                # {'covariance_type':'spherical', 'n_components':n_clusters},
                ]


    xopts = []
    for i_filename in krug_vrtnja:

        xs = []
        fs = []


        for method, param in zip(methods, params):
            i_fig = 1
            # i_filename = krug_vrtnja[0]
            dists = []
            fopts = []
            if "Birch" in method or "Agglom" in method: 
                rep = 5
            else:
                rep = 1
            for ponovi in xrange(rep):
                clustering = Clustering(path, outpath, i_filename, 
                                        method=method, xtra = param)
                start_time = time.time()
                clustering.do_cluster()
                print "duration: ", time.time()-start_time
                clustering.add_missing_values()


                optim = Fminsearch()
                optim.load_set(clustering.centres_q)
                p1 = [0,0,0]
                r1 = np.eye(3)

                xopt, fopt = optim.do_optimise_pos(p1,r1)


                # xs.append([xopt])
                # fs.append([fopt])

                dists.append(np.linalg.norm(benchmark - np.asarray(xopt)))
                fopts.append(fopt) 
                xopts.append(xopt)

            xs.append(np.mean(dists))
            fs.append(np.mean(fopts))
            

        xplot = []
        yplot = []

        i = 0
        for method, param, x in zip(methods, params, xs):

            i = i + 1
            xplot.append(i)
            # print np.linalg.norm(benchmark - np.asarray(x))
            # yplot.append(np.linalg.norm(benchmark - np.asarray(x)))
            yplot.append(x)

        fig = plt.figure(i_fig)
        plt.subplot(211)
        plt.scatter(xplot, yplot)
        plt.grid("on")
        plt.title('dist to bench')

        plt.subplot(212)
        plt.scatter(xplot, fs)
        plt.grid("on")
        plt.title('f_opt')
        
    plt.show()

