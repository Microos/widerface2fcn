import h5py
from skimage import io
import shutil, os
import random
import scipy.io as sio
import time
import numpy as np
from ProgressBar import *
import os.path as osp
from scipy.io import *



def read_wider_mat(WIDER_ROOT, SET_TYPE, MAT):
    # return imgname, WHC, bboxes
    mat = h5py.File(MAT, 'r')
    filenames = []
    WHC = [] #width, height, channel
    img_bboxes = []
    discard_num = 0
    file_list = mat['file_list'][:]
    event_list = mat['event_list'][:]
    bbx_list = mat['face_bbx_list'][:]
    print 'Reading wider mat files...'
    pbar = ProgressBar(file_list.size)
    for i in range(file_list.size):
            pbar+=1
            file_list_sub = mat[file_list[0,i]][:]
            bbx_list_sub = mat[bbx_list[0, i]][:]
            event_value = ''.join(chr(x) for x in mat[event_list[0,i]][:])
            for j in range(file_list_sub.size):
                
                root = osp.join(WIDER_ROOT,SET_TYPE,'images',event_value)
                filename = osp.join(root , ''.join([chr(x) for x in mat[file_list_sub[0, j]][:]])+'.jpg')
                im = io.imread(filename)
                bboxes = mat[bbx_list_sub[0, j]][:]
                bboxes = bbox_filter(bboxes, filename)
                if(bboxes.shape[1] == 0):
                   # print '[+] Discard:',filename
                    discard_num += 1
                    continue
                filenames.append(filename)
                WHC.append(np.array([im.shape[1], im.shape[0], im.shape[2]]))
                img_bboxes.append(bboxes.T)
    print '\n[+] Dicard {} bad images'.format(discard_num)
    del im
    return filenames, WHC, img_bboxes
    
def bbox_filter(bboxes, fname):
    # read: 'bboxes' of image 'fname'
    # do: 1. filter out any problematic bboxes
    #     2. save a fig if these are any problematic bboxes
    # return: correct bboxes
    sanity_indces = []
    problm_indces = []
    #filtering
    bboxes = np.round(bboxes).astype(np.int)
    for i in range(bboxes.shape[1]):
        b = bboxes[:,i]
        x = b[0]
        y = b[1]
        w = b[2]
        h = b[3]
        if(w*h <= 0.0):
            problm_indces.append(i)
        else:
            sanity_indces.append(i)
    return bboxes[:, sanity_indces]

def gen_label_image(whc,img_bboxes):
    label = np.zeros((whc[1], whc[0]))
    for b in img_bboxes:
        label[b[1]:b[1]+b[3],b[0]:b[0]+b[2]] = 1
    return label[np.newaxis,...]


def mk_dir_tree(build_tree_at):
    assert osp.isdir(build_tree_at)
    
    dataset_root = osp.join(build_tree_at, 'wider')
    if(FORCE_OVERWRITE): assert not osp.isdir(dataset_root)
    img_dir = osp.join(dataset_root,'img')
    cls_dir = osp.join(dataset_root,'cls')
    if shutil.os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
    shutil.os.makedirs(img_dir)
    shutil.os.makedirs(cls_dir)
    return dataset_root, img_dir, cls_dir

def convert(fcn_data_dir, wider_root,  wider_mat_dir):
    dataset_root, img_dir, cls_dir = mk_dir_tree(fcn_data_dir)
    
    mats = ['wider_face_val.mat','wider_face_train.mat']
    set_types = ['WIDER_val','WIDER_train']
    for mat, set_type in zip(mats, set_types):
        print "[{}]".format(set_type)
        
        wider_mat = osp.join(wider_mat_dir,mat)
        filenames,WHC,img_bboxes = read_wider_mat(wider_root, set_type, wider_mat)
#         lab_imgs =gen_label_image(filenames,WHC,img_bboxes)
#         del WHC, img_bboxes
        txt_content = []
        print "Writting..."
        pbar = ProgressBar(len(filenames))
        for img_path, whc, bboxes in zip(filenames, WHC, img_bboxes):
            pbar+=1
            #cp image
            short_name = img_path.split('/')[-1]
            non_ext_name = short_name.split('.')[0]
            shutil.copyfile(img_path,  osp.join(img_dir,short_name))
            #os.system('cp {} {}'.format(img_path, osp.join(img_dir,short_name)))
            
            #save mat
            label = gen_label_image(whc, bboxes) #to read: label = mat['GTcls']
            mat_path = osp.join(cls_dir,non_ext_name+'.mat')
            savemat(mat_path,{'GTcls':label})
            
            #append to txt_content
            txt_content.append(non_ext_name)
            
        #write .txt
        if 'train' in set_type:
            txtname = osp.join(dataset_root,'train.txt')
        if 'val' in set_type:
            txtname = osp.join(dataset_root,'val.txt')
        with open(txtname, 'w') as f:
            f.write('\n'.join(txt_content))
        del filenames,WHC,img_bboxes
        print '\n'


if __name__ == "__main__":

	FORCE_OVERWRITE = True #if False, when the dir `$fcn.berkeleyvision.org/data/wider` exists, the program will abort instead of cleaning up the dir.
	fcn_data_dir = '/home/rick/Space/clone/fcn.berkeleyvision.org/data' #point to your '$fcn.berkeleyvision.org/data'
	wider_root = '/home/rick/Documents/Models/WIDER_FACE/unzips70' #point to your wider dir which contains `WIDER_train` & `WIDER_val` folders
	wider_mat_dir = '/home/rick/Documents/Models/WIDER_FACE/unzips/Annotations' #point to your wider mat file dir which contains `wider_face_train.mat` & `wider_face_val.mat`
	convert(fcn_data_dir,wider_root, wider_mat_dir) #fire



