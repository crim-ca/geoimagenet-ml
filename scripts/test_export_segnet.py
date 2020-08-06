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
    task_detection = {
        "type": "thelper.tasks.Segmentation",
        "params": {
            # correspond aux deux dossiers d'images du dataset
            "class_names": ["Background", "AgriculturalLand", "BarrenLand", "ForestLand", "RangeLand", "UrbanLand", "Water"],
            # "class_names": ["AgriculturalLand":1, "BarrenLand":2, "ForestLand":3, "RangeLand":4, "UrbanLand":5, "Water":6],
            # "class_names":   {"221":0,"222":1, "251":2, "232":3, "228":4, "199":5, "238":6},
            "input_key": "image",
            'dontcare' : 255,
            "label_map_key": "label"
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
    task_regression = {
        "type": "thelper.tasks.Regression",
        "params": {
            "input_key": "image", "target_key": "1"
        }
    }


    datasets_config = {
        # loader for the test data
        "deepglobe_test": {
            "type": "thelper.data.ImageDataset",
            # "type": "thelper.data.SegmentationDataset",
            "params": {"root": "/home/sfoucher/DEV/geoimagenet/dataset_test/deepglobe_seg/test",
                       # "class_names": ["AgriculturalLand", "BarrenLand", "ForestLand", "RangeLand", "UrbanLand", "Water"],
                        # "input_key": "image",
                        #'dontcare' : 255,
                        # "label_map_key": "label"
                       "image_key": "image"
                       },
            "task": task_detection
        }
    }

    datasets_config = {
        # loader for the test data
        "deepglobe_test": {
            "type": "thelper.data.geo.ImageFolderGDataset",
            # "type": "thelper.data.SegmentationDataset",
            "params": {"root": "/home/sfoucher/DEV/geoimagenet/dataset_test/deepglobe_classif",
                       # "class_names": ["AgriculturalLand", "BarrenLand", "ForestLand", "RangeLand", "UrbanLand", "Water"],
                       # "input_key": "image",
                       # 'dontcare' : 255,
                       # "label_map_key": "label"
                       "image_key": "image",
                       "channels": [1, 2, 3]

                       },
            "task": task_detection
        }
    }
    datasets_ucmerced_config = {
        # loader for the test data
        "planes_vs_vehicles_test": {
            #        "type": "torchvision.datasets.ImageFolder",
            "type": "thelper.data.ImageFolderDataset",
            "params": {"root": "/content/Samples"},
            "task": task_detection
        }
    }



    loaders_config = {
        "batch_size": 1,
        # ci-dessous, la liste des transformations à appliquer à toutes les images...
        "base_transforms": [
            # this is necessary because the training used PIL which output images in a BGR order
            {
                "operation": "thelper.transforms.ToNumpy",
                "params": {"reorder_bgr": True},
            },
            # this is necessary because torchvision transforms assumes images as PIL objects
            {
                "operation": "torchvision.transforms.ToPILImage",
            },
            {
                "operation": "torchvision.transforms.CenterCrop",
                "params": {"size": 128},
            },
            {
                "operation": "torchvision.transforms.ToTensor",
            },
            {
                "operation": "torchvision.transforms.Normalize",
                "params": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                },
            },
        ],

        "test_split": {
            "deepglobe_test": 1.0
        },
    }
    loaders_config = {
        "batch_size": 1,
        # ci-dessous, la liste des transformations à appliquer à toutes les images...
        "base_transforms": [
            # this is necessary because the training used PIL which output images in a BGR order
            {
                "operation": "thelper.transforms.ToNumpy",
                "params": {"reorder_bgr": True},
            },
            {
                "operation": "thelper.transforms.CenterCrop",
                "params": {"size": 128},
            },
            {
                "operation": "thelper.transforms.NormalizeMinMax",
                "params": {"min": 0.0, "max": 255.0},
            },
            {
                "operation": "thelper.transforms.NormalizeZeroMeanUnitVar",
                "params": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                },
            },
            {
                "operation": "thelper.transforms.Transpose",
                "params": {
                    "axes": [2,0,1]
                },
            },
        ],

        "test_split": {
            "deepglobe_test": 1.0
        },
    }

    loaders_ucmerced_config = {
        "batch_size": 1,
        # ci-dessous, la liste des transformations à appliquer à toutes les images...
        "base_transforms": [
            {
                "operation": "torchvision.transforms.ToTensor"
            }
        ],

        "test_split": {
            "planes_vs_vehicles_test": 1.0
        },
    }
    data_config = {"datasets": datasets_config, "loaders": loaders_config}  # for Pleiade

    # task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(data_config, save_dir=save_dir)
    model_config = {
        "type": thelper.nn.gin.EncoderDecoderNet,
        "params": {"num_classes": 7, "enc_type": 'resnet18', "dec_type": 'unet_scse', "num_filters": 16,
                   "pretrained": True},
        # "state_dict": '/home/sfoucher/DEV/pytorch-segmentation/model/my_deepglobe_unet_res18_scse_2020-07-14 14:54:46.314406/model.pth',
        'state_dict': '/home/sfoucher/DEV/pytorch-segmentation/model/my_deepglobe_unet_res18_scse_2020-07-20 16:09:44.850263/model.pth',
        "task": task_detection

    }


    config = {"name": 'deepglobe-unet', "model": model_config, "datasets": datasets_config, "loaders": loaders_config,
              "trainer": trainer_config}
    # model = thelper.nn.create_model(config, task, save_dir=save_dir)
    # model.eval()
    thelper.cli.export_model(config, '/home/sfoucher/DEV/geoimagenet/pth')

    class_mapping = [('Background', 999), ('AgriculturalLand', 223), ('BarrenLand', 252), ('ForestLand', 233), ('RangeLand', 229), ('UrbanLand', 201), ('Water', 239)]

    update_model_class_mapping(class_mapping,
                               '/home/sfoucher/DEV/geoimagenet/pth/deepglobe-unet/deepglobe-unet.export.pth', None)
    # success, model, buffer, exception = load_model('/home/sfoucher/DEV/geoimagenet/pth/deepglobe-unet/deepglobe-unet.export.pth')
    # validate_model(model)


if __name__ == "__main__":
    process()
