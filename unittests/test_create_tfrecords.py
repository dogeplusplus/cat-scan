import json
import pytest

from pathlib import Path

from create_tfrecords import split_coco_annotations


@pytest.fixture
def coco_json():
    with open(Path('unittests/assets/test_coco.json'), 'r') as f:
        coco = json.loads(f.read())
    return coco


def test_train_validation_split(coco_json):
    train, val = split_coco_annotations(coco_json, validation=0.2)
    # Check the splitting works
    assert len(train["images"]) == 8
    assert len(val["images"]) == 2

    train_image_ids = [t["id"] for t in train["images"]]
    val_image_ids = [v["id"] for v in val["images"]]

    assert frozenset(train_image_ids).isdisjoint(val_image_ids)

    train_annotation_ids = [t["image_id"] for t in train["annotations"]]
    val_annotation_ids = [v["image_id"] for v in val["annotations"]]

    assert frozenset(train_annotation_ids).isdisjoint(val_annotation_ids)
