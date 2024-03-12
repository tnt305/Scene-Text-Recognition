import os
import argparse
import yaml
import shutil
from typing import List, int
from preprocessing import extract_data_from_xml, convert_to_yolov8_format, save_data
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def arguments():
    arg = argparse.ArgumentParser(add_help= False)
    arg.add_argument("--dataset_dir", default = './datasets/SceneTrialTrain')
    arg.add_argument("--class_labels", default = ['text'], type = List)
    arg.add_argument('--save_yolo_data_dir', default = './datasets/yolo_data' ,help = 'path to yolo formatted data if not exist, then create new')
    arg.add_argument("--yolo_yaml", action = 'store_true')
    arg.add_argument("--yolo_ckpt", action = 'store_true')

    arg.add_argument("--seed", default = 0, type = int)
    arg.add_argument("val_size", default = 0.2)
    arg.add_argument("--test_size", default = 0.125)
    arg.add_argument('--is_shuffle', default = True)

    arg.add_argument('--epochs', default = 200)
    arg.add_argument('--image_size', default = 1024)

    return arg

def main(arg):
    arg =arguments()
    img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(arg.dataset_dir)
    yolov8_data = convert_to_yolov8_format(
                                        img_paths, 
                                        img_sizes, 
                                        bboxes
                                    )
    train_data, test_data = train_test_split(
                                            yolov8_data, 
                                            test_size= arg.val_size, 
                                            random_state=arg.seed,
                                            shuffle= arg.is_shuffle
                                        )
    test_data, val_data = train_test_split(
                                            test_data, 
                                            test_size= arg.test_size, 
                                            random_state= arg.seed,
                                            shuffle= arg.is_shuffle
                                        )

    os.makedirs(arg.save_yolo_data_dir, exist_ok= True)

    for data_type in ['train', 'val', 'test']:
        save_dir = os.path.join(arg.save_yolo_data_dir, data_type)
        save_data(train_data if data_type == 'train' else (val_data if data_type == 'val' else test_data), arg.dataset_dir, save_dir)
    
    # create yaml
    data_yaml = {
                    'path': 'yolo_data',
                    'train': 'train/images',
                    'test': 'test/images',
                    'val': 'val/images',
                    'nc': 1,
                    'names': arg.class_labels
                }
    yolo_yaml_path = os.path.join(
                                arg.save_yolo_data_dir,
                                'data.yml'
                                )
    with open(yolo_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)


    model  = YOLO(arg.yolo_yaml).load(arg.yolo_ckpt)
    model.train(
                            data=yolo_yaml_path, 
                            epochs=arg.epochs, 
                            imgsz= arg.image_size,
                            project='models',
                            name='yolov8/detect/train'
                        )
    
if __name__ == "__main__":
    arg  = arguments()
    main(arg)
