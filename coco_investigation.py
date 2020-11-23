import json

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pathlib import Path
import imageio

data_dir = 'data'
data_type = 'train2017'
images_path = Path(data_dir) / data_type
annotation_path = Path(data_dir) / 'annotations2017' / f'cats_{data_type}.json'

coco = COCO(annotation_path)

cats = coco.loadCats(coco.getCatIds())
names = [cat['name'] for cat in cats]
print(f'COCO categories: \n{" ".join(names)}\n')

super_cats = set([cat['supercategory'] for cat in cats])
print(f'COCO supercategories: \n{" ".join(super_cats)}\n')

cat_ids = coco.getCatIds(catNms=['cat'])
image_ids = coco.getImgIds(catIds=cat_ids)
img = coco.loadImgs(image_ids[0])[0]

x = imageio.imread(img['coco_url'])


# plt.axis('off')
# plt.imshow(x)
annotation_ids = coco.getAnnIds(imgIds=image_ids, catIds=cat_ids, iscrowd=None)
annotations = coco.loadAnns(annotation_ids)

# coco.showAnns(annotations)
# plt.show()

with open(annotation_path, 'r') as f:
    annotations_json = json.loads(f.read())
print(annotations_json.keys())
