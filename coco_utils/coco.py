import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from PIL import ImageOps
from fullgrad import misc_functions
from fullgrad.fullgrad import FullGrad
from matplotlib import pyplot as plt

import os
import pdb
import torch
import time
import json
from statistics import mean

class CocoDataset():

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    
    CLASSES_sorted = np.sort(CLASSES)

    def __init__(self, PATH, set_name, extract_saliency):
        
        # create folders
        self.path = PATH
        self.part_info = []
        # open a new list for annotations.
        self.size = 224
        self.set_name = set_name
        self.extract_saliency = False
        self.model = None
        self.data_path = os.path.join(self.path, 'dataset', 'coco')
        self.processed_path = os.path.join(self.data_path, \
        								   'fine_tune', self.set_name[:-4])
        self.create_folders(self.processed_path)
        # load annotations
        ann_filename = 'annotations/instances_' + set_name + '.json'
        ann_file = os.path.join(self.data_path, ann_filename)
        self.img_infos = self.load_annotations(ann_file) 
        if extract_saliency:
            self.extract_saliency = extract_saliency
            # if saliency is to be extracted, create modified ann_files.
            ann_filename_saliency = 'dataset/coco/annotations/instances_' + set_name + '_wParts.json'
            self.ann_file_saliency = os.path.join(PATH, ann_filename_saliency)
            # initialize new dataset instances
            self.initialize_annotations()
            # construct full-grad object
            model_path = os.path.join(self.path, 'models', \
                                      'vgg16bn_fromscratch_90_best.pth')
            self.model = misc_functions.init_model(model_path)        
            self.device = "cpu"
            self.fullgrad = FullGrad(self.model, self.device) 
    
    def initialize_annotations(self):
        # copies unchanged instances of old dataset
        # to new dataset.
        self.info_ = self.coco.dataset['info']
        self.categories_ = self.coco.dataset['categories']
        self.licenses_ = self.coco.dataset['licenses']
        self.images_ = self.coco.dataset['images']
        # initialize an empty list for new annotation file.
        self.annotations_ = [] 

    def create_folders(self,path):
        # create folders if not exist
        for idx, cname in enumerate(self.CLASSES):
            class_folder = os.path.join(path, cname)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
        return

    def convert_boxes(self, box):
        area = (box[0], box[1], box[2], box[3])
        measure = box[2] * box[3]
        return area, measure

    def seperate_parts(self, img, bboxes, labels, integral_flags, debug=True):
        partwlabel = []
        integral_list = []

        for idx, box in enumerate(bboxes):
            area, measure = self.convert_boxes(box)
            part = img.crop(area)
            if self.extract_saliency:
                # extract saliency maps of each part in image
                
                # transform images
                #part.show()
                transforms = misc_functions.get_transforms()
                part_ = transforms(part)
                part_ = part_.unsqueeze(0) 
                # convert labels to sorted labels
                labels_ = misc_functions.get_indices(labels, self.CLASSES, \
                                                    self.CLASSES_sorted)
                # iterate over converted labels
                with torch.no_grad():
                    self.model.eval()
                    raw_output = self.model(part_)
                    probs = torch.softmax(raw_output[0], dim=0)
                    # one should convert coco annotations into alphabetical order
                    # to get desired class probability.
                        
                    #print("Desired Class Probability:", probs[labels_[idx]])
                    #print("Predicted class and probability:", \
                    #       self.CLASSES_sorted[torch.argmax(probs)],\
                    #       torch.max(probs))
                cam = self.fullgrad.saliency(part_, \
                                             target_class=torch.tensor([labels_[idx]]))
                part_inversed = misc_functions.get_transforms_inverse(part_[0,:,:,:].cpu())
                saliency_map = misc_functions.save_saliency_map(part_inversed, cam[0,:,:,:], \
                                                                './dummy.jpg')
                saliency_map = torch.from_numpy((saliency_map / \
                                                np.sum(saliency_map))*\
                                                (self.size*self.size)).\
                                                type(torch.DoubleTensor).to(self.device)

                integral_saliency_map = misc_functions.integral_image_compute(saliency_map, \
                                                                              1, self.size, self.size, \
                                                                              device = self.device).\
                                                                              squeeze()
                if integral_flags[idx] == False:
                    integral_list.append(torch.tensor([]))
                else:
                    integral_list.append(integral_saliency_map)        
            if debug:
                part.show()
                print(self.CLASSES[labels[idx]-1])
            
            partwlabel.append([part, labels[idx], measure])

        return partwlabel,integral_list

    def read_image(self, path):
        return Image.open(path)

    def save_anns(self):
        print("Saving annotations with parts...")
        json_data = {
                'info': self.info_,
                'licenses': self.licenses_,
                'images': self.images_,
                'annotations': self.annotations_,
                'categories': self.categories_
                }
        with open(self.ann_file_saliency, 'w') as fp:
            json.dump(json_data, fp, sort_keys =True, indent=4)
        print("Done, PATH: {}".format(self.ann_file_saliency))
    def process_images(self, save=False):
        # filter images
        valid_inds = self._filter_imgs()
        self.img_infos = [self.img_infos[i] for i in valid_inds]
        imgs_w_parts = []
        # load images and metas.
        times = []
        #len(self.img_infos))
        for idx in range(0, len(self.img_infos)):
            # read img info, annotations.
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
            #print("Check ID: {}".format(img_info['id']))
            # get bboxes and labels
            bboxes = ann_info['bboxes']
            labels = ann_info['labels']
            integral_flags = ann_info['write_integral']
            # read image
            img = self.read_image(os.path.join(self.data_path, self.set_name, img_info['filename']))
                       
            # pass to crop function
            start = time.time()
            part_tuples, integrals = self.seperate_parts(img, bboxes, labels, \
                                                         integral_flags, debug=False)
            #print("Len ann info: {}".format(len(labels)))
            #print("Number of Integrals: {}".format(len(integrals)))
            self.gather_annotations(idx, integrals)
            end = time.time()
            times.append(end-start)
            scales = [element[2]/(img_info['height'] * img_info['width']) for element in part_tuples]
            self.part_info.append([len(part_tuples), scales])
            if idx % 1 == 0:
                mean_time = mean(times)
                print("Image count:[{}] / [{}], Total Parts: {} Mean Time: {:.4f} sec.\n".\
                       format(idx, len(self.img_infos),len(part_tuples), mean_time))
                times = []
            
            # collect tuples to save parts w/labels.
            if save:
                for tup_idx, part_tuple in enumerate(part_tuples):
                    save_path = os.path.join(self.processed_path, self.CLASSES[part_tuple[1]-1])
                    im_name = str(img_info['id']) + '_' + str(tup_idx) + '.jpg'
                    save_name = os.path.join(save_path, im_name)
                    # ignore 1d images after crop
                    if 0 in part_tuple[0].size:
                        continue
                    else:
                        part_tuple[0].save(save_name)
        self.save_anns()
        return self.part_info
                
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def gather_annotations(self, idx, integrals):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        for counter, part_info in enumerate(ann_info):
            part_info['gt_saliency_map'] = integrals[counter].numpy().tolist()
            self.annotations_.append(part_info)

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        write_integral = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_bboxes.append(bbox)
                write_integral.append(False)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                write_integral.append(True)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            write_integral=write_integral)

        return ann
