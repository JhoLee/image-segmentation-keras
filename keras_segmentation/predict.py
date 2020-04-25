import glob
import random
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
from keras.models import load_model

from .train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array, DATA_LOADER_SEED, class_colors , get_pairs_from_paths
from .models.config import IMAGE_ORDERING
from . import metrics

import six

import matplotlib.pyplot as plt




random.seed(DATA_LOADER_SEED)

def model_from_checkpoint_path(checkpoints_path):

    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model

def blur3(image, mask):
    ori_w = image.shape[0]
    ori_h = image.shape[1]
    
    
    out_w = mask.shape[0]
    out_h = mask.shape[1]
    
    image_ = np.asarray(image, dtype=np.uint8)
    mask_ = np.asarray(mask, dtype=np.uint8)
        
    blur = cv2.blur(image_, (25, 25)) #  ori_w X ori_h
    
#     print(type(image), type(image_), type(mask), type(mask_), type(blur))
    #TODO
#     print(mask_.shape)
    mask_ = cv2.resize(mask_, (ori_h, ori_w)) # ori_w, ori_h
    mask_ = mask_[:, :, np.newaxis]
#     print(mask_.shape)
    if mask_.shape[-1] > 0:
        mask_ = (np.sum(mask_, -1, keepdims=True) >=1)
        mask_ = (np.sum(mask_, -1, keepdims=True) >=1)
#         print(mask_.shape)
        after_blurring = np.where(mask_, blur, image_).astype(np.uint8)
    else:
        after_blurring = image_.astype(np.uint8)
    return image_, mask_, blur, after_blurring

def blur(image, mask):
    ori_w = image.shape[0]
    ori_h = image.shape[1]
    
    
    out_w = mask.shape[0]
    out_h = mask.shape[1]
    
    image_ = np.asarray(image, dtype=np.uint8)
    mask_ = np.asarray(mask, dtype=np.uint8)
    
    factor = 3.0
    
    kW = int(ori_w / factor)
    kH = int(ori_h / factor)
    
    if kW % 2 == 0:
        kW -= 1
    if kH % 2 == 0:
        kH -= 1
    
    blur = cv2.GaussianBlur(image_, (kW, kH), 0) #  ori_w X ori_h
    
#     print(type(image), type(image_), type(mask), type(mask_), type(blur))
    #TODO
    
    mask_ = cv2.resize(mask_, (ori_h, ori_w)) # ori_w, ori_h
        
    if mask_.shape[-1] > 0:
        mask_ = (np.sum(mask_, -1, keepdims=True) >=1)
        after_blurring = np.where(mask_, blur, image_).astype(np.uint8)
    else:
        after_blurring = blur.astype(np.uint8)
    return after_blurring

    
def predict(model=None, inp=None, out_fname=None, blur_fname=None, checkpoints_path=None, original_size=(0,0), save_blur=False):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    if checkpoints_path is not None:
        model.load_weights(checkpoints_path)


    assert (inp is not None)
    assert((type(inp) is np.ndarray) or isinstance(inp, six.string_types)
           ), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    #TODO

   
    seg_img = np.zeros((output_height, output_width, 3))
    if blur_fname is not None:
#         inp_ = cv2.resize(inp, (output_width, output_height))
        pr_ = np.zeros((output_height, output_width, 1))
        pr_ = np.where(pr==1, 1, 0)
#         blur_img = blur(inp, pr_)
#         print(np.unique(pr))
#         print(np.unique(pr_))
        _, _, _, blur_img = blur3(inp, pr_)
        
    
    
#     colors = class_colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c)*(colors[c][2])).astype('uint8')
    
    
    seg_img2 = cv2.resize(seg_img, (orininal_h, orininal_w), interpolation=cv2.INTER_NEAREST)
    seg_img = cv2.resize(seg_img, (orininal_h, orininal_w))
#     blur_img = cv2.resize(blur_img, (original_w, original_h)) 
    

    if out_fname is not None:
#         cv2.imwrite(out_fname, seg_img)
#         cv2.imwrite(out_fname.replace('.jpg', '__.jpg'), seg_img2)
        cv2.imwrite(out_fname, seg_img2)
    if blur_fname is not None:
        cv2.imwrite(blur_fname, blur_img)

    return pr, blur


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, blur_dir=None, save_blur=False):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))

    assert type(inps) is list

    all_prs = []
    all_blurs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        if blur_dir is None:
            blur_fname = None
        else:
            if isinstance(inp, six.string_types):
                blur_fname = os.path.join(blur_dir, os.path.basename(inp))
            else:
                blur_fname = os.path.join(blur_dir, str(i) + ".jpg")
        pr, blurs = predict(model, inp, out_fname, blur_fname, save_blur=save_blur)
        all_prs.append(pr)
        all_blurs.append(blurs)

    return all_prs, blurs

def blur_multiple(imgs=None, img_dir=None, inps=None, inp_dir=None, out_dir=None, save=False):
    if imgs is None and (img_dir is not None):
        imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
        
    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + \
                glob.glob(os.path.join(inp_dir, "*.bmp"))
    
    assert type(imgs) is list
    assert type(inps) is list
    
    all_blurs = []
    
    for i, img, inp in enumerate(tqdm(zip(imgs, inps))):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")
        
        blur = blur2(img, inp, out_fname)
        all_blurs.append(blur)
    
    return all_blurs
                                 

def evaluate( model=None , inp_images=None , annotations=None, inp_images_dir=None ,annotations_dir=None , checkpoints_path=None ):
    
    if model is None:
        assert (checkpoints_path is not None) , "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)
        
    if inp_images is None:
        assert (inp_images_dir is not None) , "Please privide inp_images or inp_images_dir"
        assert (annotations_dir is not None) , "Please privide inp_images or inp_images_dir"
        
        paths = get_pairs_from_paths(inp_images_dir , annotations_dir )
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])
        
    assert type(inp_images) is list
    assert type(annotations) is list
        
    tp = np.zeros( model.n_classes  )
    fp = np.zeros( model.n_classes  )
    fn = np.zeros( model.n_classes  )
    tn = np.zeros( model.n_classes  )
    n_pixels = np.zeros( model.n_classes  )
    
    for inp , ann   in tqdm( zip( inp_images , annotations )):
        pr, _ = predict(model , inp )
        gt = get_segmentation_array( ann , model.n_classes ,  model.output_width , model.output_height , no_reshape=True  )
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
                
        for cl_i in range(model.n_classes ):
            
            tp[ cl_i ] += np.sum( (pr == cl_i) * (gt == cl_i) )
            fp[ cl_i ] += np.sum( (pr == cl_i) * ((gt != cl_i)) )
            fn[ cl_i ] += np.sum( (pr != cl_i) * ((gt == cl_i)) )
            tn[ cl_i ] += np.sum( (pr != cl_i) * ((gt != cl_i)) )
            n_pixels[ cl_i ] += np.sum( gt == cl_i  )
            
    cl_wise_score = tp / ( tp + fp + fn ) if (tp + fp + fn) is not 0 else 0
    precision = tp / ( tp + fn ) if (tp + fn) is not 0 else 0
    recall = tp / (tp + fp) if (tp + fn) is not 0 else 0
    
    f1_score = 2 * ( precision * recall ) / (precision + recall ) if (precision + recall) is not 0 else 0
    
    n_pixels_norm = n_pixels /  np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
    
    return {"frequency_weighted_IU":frequency_weighted_IU , "mean_IU":mean_IU , "class_wise_IU(IoU)":cl_wise_score, "f1_score": f1_score, "acc": accuracy }


