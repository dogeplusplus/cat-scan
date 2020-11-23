import csv
import tqdm
import requests
from pycocotools.coco import COCO

coco = COCO('data/annotations2017/cats_train2017.json')
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['cat'])
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)
print("imgIds: ", imgIds)
print("images: ", images)

for im in tqdm.tqdm(images):
    img_data = requests.get(im['coco_url']).content
    with open('data/coco_cats/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)

