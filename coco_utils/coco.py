import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from PIL import ImageOps

import os
import pdb


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
    def __init__(self, PATH, set_name, extract_saliency):
        
        # create folders
        self.path = PATH
        self.part_info = []
        self.size = 224, 224
        self.set_name = set_name
        self.processed_path = os.path.join(PATH, 'fine_tune', self.set_name[:-4])
        self.create_folders(self.processed_path)
        if extract_saliency:
        	# if saliency is to be extracted, create modified ann_files,
        	# full-grad object.
        	ann_filename_saliency = 'annotations/instances' + set_name \
        							+ '_wParts.json'
        	ann_file = os.path.join(PATH, ann_filename_saliency)
        	
        
        
        # get ann file.
        ann_filename = 'annotations/instances_' + set_name + '.json'
        ann_file = os.path.join(PATH, ann_filename)
        self.img_infos = self.load_annotations(ann_file)
        
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

    def seperate_parts(self, img, bboxes, labels, debug=True):
        partwlabel = []
        for idx, box in enumerate(bboxes):
            area, measure = self.convert_boxes(box)
            part = img.crop(area)
            if debug:
                pdb.set_trace()
                part.show()
                print(self.CLASSES[labels[idx]-1])
            partwlabel.append([part, labels[idx], measure])
        return partwlabel

    def read_image(self, path):
        return Image.open(path)

    def process_images(self, extract_saliency, save=True):
        # filter images
        valid_inds = self._filter_imgs()
        self.img_infos = [self.img_infos[i] for i in valid_inds]
        imgs_w_parts = []
        # load images and metas.
        for idx in range(0, len(self.img_infos)):
            # read img info, annotations.
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
            # get bboxes and labels
            bboxes = ann_info['bboxes']
            labels = ann_info['labels']
            # read image
            img = self.read_image(os.path.join(self.path, self.set_name, img_info['filename']))
                       
            # pass to crop function
            part_tuples = self.seperate_parts(img, bboxes, labels, debug=True)
            scales = [element[2]/(img_info['height'] * img_info['width']) for element in part_tuples]
            self.part_info.append([len(part_tuples), scales])
            if idx % 100 == 0:
                print("Image count:[{}]/[{}], Total Parts:{}\n".\
                       format(idx, len(self.img_infos),len(part_tuples)))
 
            
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


    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        #pdb.set_trace()
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

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

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
            seg_map=seg_map)

        return ann
