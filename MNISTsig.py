import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import esig
from esig import tosig as ts
import sys
import os
from sklearn.preprocessing import MinMaxScaler as sc
import math

#signature computed up to level
level=3
#number of subdivision passes
fractal_depth=2
#number of sampled points
sample_size=3
#size of data
signature_dim=(pow(2,level+1)-1)
#number of paths in the dyadic tree
#if training only with lowest level of dyadic samples set
sampled_pieces=2**fractal_depth
#sampled_pieces=(pow(2,fractal_depth+1)-1)

#directory path
parent_dir='/Users/user/Google Drive/atom programming/mnistseq/'
#data_folder
data_dir=os.path.join(parent_dir,'sequences/')

#adds noise to path
def add_noise(path):
    noise=np.floor(np.random.normal(0,1,path.shape))
    return path+noise

#returns tree of dyadic subpaths up to depth and normalized to (0,1)
def dyadic_paths(path, depth, noise=False):
    length=path.shape[0]
    path_tree=[]
    path_tree.append(path)
    min_part_length=4


    for i in range(1,min(depth+1,math.floor(length/min_part_length)+1)):
        parent_index=len(path_tree)-2**(i-1)

        for i in range(parent_index,parent_index+2**(i-1)):
            path_tree.append(path_tree[i][0:math.floor(len(path_tree[i])/2)])
            path_tree.append(path_tree[i][math.floor(len(path_tree[i])/2):len(path_tree[i])])

    #normalize data
    sctrain=sc(feature_range=(0,1))
    sctrain.fit(path)

    if noise==True:
        for i,y in enumerate(path_tree):
            path_tree[i]=add_noise(y)

    for i,y in enumerate(path_tree):
        path_tree[i]=sctrain.transform(y)

    return path_tree

#samples "sample_size" points equidistantly from each path entry in the tree
#with "path" in the root and it's dyadic sub-paths up to "depth".
def dyadic_sample(path, depth, sample_size, fractal_sample=True):
    d_paths=dyadic_paths(path,depth)
    sampled_paths=[]

    for x in d_paths:
        step_size=math.floor(len(x)/(sample_size-1))
        sample=[]
        for j in range(0,sample_size-1):
            sample.append(x[j*step_size])

        sample.append(x[-1])
        sampled_paths.append(sample)

    return np.array(sampled_paths,dtype='float')

#returns a list of signatures of fractally sampled paths out of "path" at given level
def frac_sig(path, sig_level, frac_depth, sample_size,just_leaves=False):
    sampled_paths=dyadic_sample(path,frac_depth, sample_size)

    fractured_signatures=[]
    if just_leaves==True:
        a=-2**fractal_depth
    else:
        a=0
    for y in sampled_paths[a:]:
        fractured_signatures.append(esig.stream2sig(y,sig_level))

    return np.array(fractured_signatures)

def frac_seq_sig(path, sig_level, frac_depth, just_leaves=False):
    sampled_paths=dyadic_paths(path,frac_depth)

    fractured_signatures=[]
    if just_leaves==True:
        a=-2**fractal_depth
    else:
        a=0
    for x in sampled_paths[a:]:
        fractured_signatures.append(esig.stream2sig(x,sig_level))

    return np.array(fractured_signatures)

#tests methods on sample path
def test_path(path_length):
    test_path=[]
    for i in range(path_length):
        test_path.append([i,i])

    paths=dyadic_paths(np.array(test_path),fractal_depth)
    print(np.shape(paths))
    print(paths)

    spaths=dyadic_sample(np.array(test_path),fractal_depth,sample_size)
    print(np.shape(spaths))
    print(spaths)


    signatures=frac_sig(np.array(test_path),level,fractal_depth,sample_size)
    print(signatures.shape)
    print(signatures)

    psignatures=frac_seq_sig(np.array(test_path),level,fractal_depth)
    print(psignatures.shape)
    print(psignatures)

# find the range of path lengths in data
def path_length_bounds(data_dir):

    min_max_path_length=[28,0]

    #locating training data files
    with os.scandir(data_dir) as entries:
        for entry in entries:

            if "points" in entry.name:

                df = pd.read_csv(os.path.join(data_dir,entry.name))
                df = df.drop(df[(df.col< 0) | (df.row < 0)].index)

                if min_max_path_length[0]>df.shape[0]:
                    min_max_path_length[0]=df.shape[0]

                if min_max_path_length[1]<df.shape[0]:
                    min_max_path_length[1]=df.shape[0]

    return min_max_path_length



#reading and processing raw data and writing it to files
def prepare_data(parent_dir, data_dir,save_dir_name,fractal_depth,signature_level,sample_size,sampling=False,just_leaves=False):
    train_images=np.empty([0,sampled_pieces,signature_dim])
    train_labels=np.empty([0,10])
    test_images=np.empty([0,sampled_pieces,signature_dim])
    test_labels=np.empty([0,10])

    with os.scandir(data_dir) as entries:
        for entry in entries:

            #locating training data files
            if "points" in entry.name and "train" in entry.name:
                    a=entry.name.split('-')
                    label_filename="trainimg-"+a[1]+"-targetdata.txt"

                    dflabels=pd.read_csv(os.path.join(data_dir,label_filename), sep=' ',names=range(0,14))
                    x=dflabels.iloc[:1,0:10].to_numpy(dtype='int')
                    train_labels=np.append(train_labels,x,axis=0)

                    df = pd.read_csv(os.path.join(data_dir,entry.name))
                    df = df.drop(df[(df.col< 0) | (df.row < 0)].index)
                    y = df.to_numpy(dtype='float')

                    if sampling==False:
                        train_images= np.append(train_images,np.expand_dims(frac_seq_sig(y,signature_level,fractal_depth,just_leaves),axis=0),axis=0)
                    else:
                        train_images= np.append(train_images,np.expand_dims(frac_sig(y,signature_level,fractal_depth,sample_size,just_leaves),axis=0),axis=0)


            #locating test data files
            if "points" in entry.name and "test" in entry.name:
                    a=entry.name.split('-')
                    label_filename="testimg-"+a[1]+"-targetdata.txt"
                    dflabels=pd.read_csv(data_dir+label_filename, sep=' ',names=range(0,14))
                    x=dflabels.iloc[:1,0:10].to_numpy(dtype='int')
                    test_labels=np.append(test_labels,x,axis=0)
                    df = pd.read_csv(data_dir+entry.name)
                    df = df.drop(df[(df.col< 0) | (df.row < 0)].index)
                    y = df.to_numpy(dtype='float')

                    if sampling==False:
                        test_images= np.append(test_images,np.expand_dims(frac_seq_sig(y,signature_level,fractal_depth,just_leaves),axis=0),axis=0)
                    else:
                        test_images= np.append(test_images,np.expand_dims(frac_sig(y,signature_level,fractal_depth, sample_size,just_leaves),axis=0),axis=0)

    save_dir=os.path.join(parent_dir,save_dir_name)
    #saving signatures up to determined level in subdirectory sig_seq__frac_3
    np.save(save_dir+'/train_images.npy',train_images)
    np.save(save_dir+'/train_labels.npy',train_labels)
    np.save(save_dir+'/test_images.npy',test_images)
    np.save(save_dir+'/test_labels.npy',test_labels)


prepare_data(parent_dir, data_dir,'leaves',fractal_depth,level,sample_size,sampling=False,just_leaves=True)
