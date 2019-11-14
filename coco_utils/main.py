from coco import CocoDataset
import matplotlib.pyplot as plt
import os
import pdb
import pickle

def main(PATH, set_name, construct_dataset, extract_saliency):
    
    if construct_dataset:
    	# extracted cropped parts of COCO and
    	# save them with a new structure.
        coco_ds = CocoDataset(PATH, set_name, extract_saliency)
        dataset_info = coco_ds.process_images()
    if extract_saliency:
    	# 
    	coco_ds = CocoDataset(PATH, set_name, extract_saliency)
    	dataset_info = coco_ds.process_images()
    	
if __name__ == '__main__':
    PATH = '/home/cancam/imgworkspace/fullgrad-saliency'
    set_names = ['train2017']
    for set_name in set_names:
    	construct_dataset = False
    	extract_saliency = True
    	main(PATH, set_name, construct_dataset, extract_saliency)
