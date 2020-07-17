from geoimagenet_ml.ml.impl import (
    load_model,
    validate_model,
    create_batch_patches,
    retrieve_annotations,
    retrieve_taxonomy,
)
import os
import torch
import torchvision
import thelper

from scripts.update_model_classes import update_model_class_mapping


def process():
    bconversion = None
    if bconversion is not None:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)
        ckptdata = torch.load("ckpt.pth")['model_state_dict']

        init_keys = ckptdata.keys()
        print(init_keys)
        new_dict = {}
        for key, val in ckptdata.items():
            key = key.replace('backbone.body.layers',
                              'backbone.body')  # for some reason there are different names for the backbone
            new_dict[key] = val
        model.load_state_dict(new_dict)
        torch.save(model.state_dict(), 'model_best_finetuning_GIN_test2.pth')
        # model.load_state_dict(new_dict)

    # initialisation des logs (pour TOUT imprimer ce qui se passe)
    thelper.utils.init_logger()

    save_dir = 'sessions'
    task_config = {
                "type": "thelper.tasks.Classification",
                "params": {
                    "class_names": ["AgriculturalLand", "BarrenLand", "ForestLand", "RangeLand", "UrbanLand", "Water"],
                    "input_key": "data",
                    "label_key": "label"
                }
            }
    trainer_config = {
        "epochs": 5,
        "monitor": "accuracy",
        "optimization": {
            "loss": {
                "type": "torch.nn.CrossEntropyLoss"
            },
            "optimizer": {
                "type": "torch.optim.Adam",
                "params": {
                    "lr": 0.001
                }
            }
        },
    }

    datasets_config = {
        # loader for the test data
        "deepglobe_test": {
            "type": "thelper.data.ImageDataset",
            # "type": "thelper.data.SegmentationDataset",
            "params": {"root": "/home/sfoucher/DEV/geoimagenet/dataset_test/deepglobe_classif/val",
                       # "class_names": ["AgriculturalLand", "BarrenLand", "ForestLand", "RangeLand", "UrbanLand", "Water"],
                        # "input_key": "image",
                        # 'dontcare' : 255,
                        # "label_map_key": "label"
                       "image_key": "image"
                       },
            "task": task_config
        }
    }


    loaders_config = {
        "batch_size": 1,
        # ci-dessous, la liste des transformations à appliquer à toutes les images...
        "base_transforms": [

            {
                "operation": "thelper.transforms.Resize",
                "params": {"dsize": [224, 224]},
            },
            {
                "operation": "thelper.transforms.NormalizeMinMax",
                "params": {
                    "min": [0, 0, 0],
                    "max": [255, 255, 255]
                },
            },
            {
                "operation": "thelper.transforms.NormalizeZeroMeanUnitVar",
                "params": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                },
            },
            {
                "operation": "torchvision.transforms.ToTensor",
            },
        ],

        "test_split": {
            "deepglobe_test": 1.0
        },
    }

    data_config = {"datasets": datasets_config, "loaders": loaders_config}  # for Pleiade
    # data_config = {"datasets": datasets_ucmerced_config, "loaders": loaders_ucmerced_config} # for UCMerced (30 cms)
    # task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(data_config, save_dir=save_dir)
    model_config = {
        "type": torchvision.models.resnet18,
        "params": {"pretrained": True},
        "state_dict": '/home/sfoucher/DEV/pytorch-segmentation/model/resnet_2020-07-16 14:36:58.634014/model.pth',
        "task": task_config

    }
    export_config = {
        "ckpt_name": "test-resnet18-imagenet.pth",
        "trace_name": "test-resnet18-imagenet.zip",
        "save_raw": True,
        "trace_input": "torch.rand(1, 3, 224, 224)",
        "task": {
            "type": "thelper.tasks.Classification",
            "params": {
                "class_names": "tests/meta/imagenet_classes.json",
                "input_key": "0",
                "label_key": "1"
            }
        }
    }

    config = {"name": 'deepglobe-resnet-18', "model": model_config, "datasets": datasets_config, "loaders": loaders_config,
              "trainer": trainer_config}
    # model = thelper.nn.create_model(config, task, save_dir=save_dir)
    # model.eval()
    thelper.cli.export_model(config, '/home/sfoucher/DEV/geoimagenet/pth')

    class_mapping = [('AgriculturalLand', 223), ('BarrenLand', 252), ('ForestLand', 233), ('RangeLand', 229),
                     ('UrbanLand', 200), ('Water', 239)]

    update_model_class_mapping(class_mapping,
                               '/home/sfoucher/DEV/geoimagenet/pth/deepglobe-resnet-18/deepglobe-resnet-18.export.pth', None)
    # success, model, buffer, exception = load_model('/home/sfoucher/DEV/geoimagenet/pth/deepglobe-unet/deepglobe-unet.export.pth')
    # validate_model(model)


if __name__ == "__main__":
    process()
