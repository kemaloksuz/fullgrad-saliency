from coco import CocoDataset
import matplotlib.pyplot as plt
import os
import pdb
import pickle

def main(PATH, set_name, construct_dataset, extract_saliency):
    
    if construct_dataset:
    	# extracted cropped parts of COCO and
    	# save them with a new structure.
        coco_ds = CocoDataset(PATH, set_name)
        dataset_info = coco_ds.process_images(extract_saliency)    

if __name__ == '__main__':
    PATH = '/home/cancam/workspace/gradcam_plus_plus-pytorch/data/coco'
    set_names = ['train2017', 'val2017']
    for set_name in set_names:
    	construct_dataset = True
    	extract_saliency = False
    	main(PATH, set_name, construct_dataset, extract_saliency)
