import numpy as np
from typing import Dict, Tuple
from copy import deepcopy


def split_coco_annotations(coco_json: Dict, validation: float = 0.2) -> Tuple[Dict, Dict]:
    """ Split coco annotations into training and test for serialization later.

    Args:
        coco_json: COCO annotation json file
        validation: proportion of images to be part of the validation set

    Returns:
        JSON annotations split into train and validation images
    """
    image_ids = [image["id"] for image in coco_json["images"]]
    validation_size = int(validation * len(image_ids))
    np.random.shuffle(image_ids)
    train_images, val_images = image_ids[validation_size:], image_ids[:validation_size]

    train_coco = coco_json
    val_coco = deepcopy(coco_json)

    train_coco["images"] = [image_data for image_data in train_coco["images"] if image_data["id"] in train_images]
    train_coco["annotations"] = [ann for ann in train_coco["annotations"] if ann["image_id"] in train_images]

    val_coco["images"] = [image_data for image_data in val_coco["images"] if image_data["id"] in val_images]
    val_coco["annotations"] = [ann for ann in val_coco["annotations"] if ann["image_id"] in val_images]

    return train_coco, val_coco
