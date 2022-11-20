import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import csv
from xml.dom import minidom
from tqdm import main, tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)

classes_id_to_txt = ["speed limit 20" ,"speed limit 30" ,"speed limit 50" ,"speed limit 60" ,"speed limit 70" ,"speed limit 80" ,"restriction ends 80" ,"speed limit 100" ,"speed limit 120" ,"no overtaking" ,"no overtaking" ,"priority at next intersection" ,"priority road" ,"give way" ,"stop" ,"no traffic both ways" ,"no trucks" ,"no entry" ,"danger" ,"bend left" ,"bend right" ,"bend" ,"uneven road" ,"slippery road" ,"road narrows" ,"construction" ,"traffic signal" ,"pedestrian crossing" ,"school crossing" ,"cycles crossing" ,"snow" ,"animals" ,"restriction ends" ,"go right" ,"go left" ,"go straight" ,"go right or straight" ,"go left or straight" ,"keep right" ,"keep left" ,"roundabout" ,"restriction ends (overtaking)" ,"restriction ends (overtaking (trucks))"]

def extract_info_from_txt(txt_file, imgsz):
    info_list = []
    with open(txt_file) as fd:
        csvFile = csv.reader(fd, delimiter=";")
        for line in csvFile:
            if len(info_list) == 0 or line[0] != info_list[-1]["filename"]:
                info_dict = {}
                info_dict["filename"] = line[0]
                info_dict["imgsz"] = imgsz
                info_dict["boxes"] = []
                info_list.append(info_dict)
            info_list[-1]["boxes"].append({
                "box": tuple([int(corner) for corner in line[1:-1]]),
                "class": int(line[-1])
            })
    return info_list

def convert_to_yolo(info_dict, root_dir = ""):
    print_buffer = {}
    for d in info_dict:
        print_buffer["filename"] = d["filename"]
        imgsz = d["imgsz"]
        annotations = []
        for b in d["boxes"]:
            bb = b["box"]
            c = b["class"]
            box_center = ((bb[0] + bb[2]) / (2 * imgsz[0]), (bb[1] + bb[3]) / (2 * imgsz[1]))
            box_size = ((bb[2] - bb[0]) / imgsz[0], (bb[3] - bb[1]) / imgsz[1])

            annotation = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(c, box_center[0], box_center[1], box_size[0], box_size[1])
            annotations.append(annotation)
        save_file_name = os.path.join(root_dir, "annotations", print_buffer["filename"].replace("ppm", "txt"))
        if not os.path.exists(os.path.join(root_dir, "annotations")):
            os.mkdir(os.path.join(root_dir, "annotations"))
        print("\n".join(annotations), file= open(save_file_name, "wt"))



annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

def prepare():
    imgsz = (1360, 800)
    labels_trainfile = "images/gt.txt"
    info_list = extract_info_from_txt(labels_trainfile, imgsz)
    convert_to_yolo(info_list)

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), classes_id_to_txt[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()

def test_annotations():
    annotation_file = random.choice(annotations)
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]

    #Get the corresponding image file
    image_file = annotation_file.replace("annotations", "images").replace("txt", "ppm")
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)

    #Plot the Bounding Box
    plot_bounding_box(image, annotation_list)

def train_test_split(images, annotations, test_size, shuffle=True):
    assert test_size >= 0.0 and test_size <= 1.0, "test_size is not valid"
    items_number = len(annotations)
    if shuffle:
        random_indexes = torch.randperm(items_number).detach().cpu()
    else:
        random_indexes = torch.arange(items_number)
    test_images, train_images = [images[x] for x in random_indexes[:int(items_number * test_size)]], [images[x] for x in random_indexes[int(items_number * test_size):]]
    test_annotations, train_annotations = [annotations[x] for x in random_indexes[:int(items_number * test_size)]], [annotations[x] for x in random_indexes[int(items_number * test_size):]]
    return train_images, test_images, train_annotations, test_annotations

#Utility function to move images 
def copy_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copyfile(f, os.path.join(destination_folder, os.path.basename(f)))
        except:
            print(f)
            assert False

def split_dataset():
    # Read images and annotations
    images = [os.path.join('images', x.replace("txt", "ppm")) for x in os.listdir('annotations') if x[-3:] == "txt"]
    annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]
    dirs = ["train", "val", "test"]
    for dir in dirs:
        images_dir = os.path.join("images", dir)
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        os.mkdir(os.path.join(images_dir))
        annotations_dir = os.path.join("annotations", dir)
        if os.path.exists(annotations_dir):
            shutil.rmtree(annotations_dir)
        os.mkdir(annotations_dir)


    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5)

    copy_files_to_folder(train_images, 'images/train')
    copy_files_to_folder(val_images, 'images/val/')
    copy_files_to_folder(test_images, 'images/test/')
    copy_files_to_folder(train_annotations, 'annotations/train/')
    copy_files_to_folder(val_annotations, 'annotations/val/')
    copy_files_to_folder(test_annotations, 'annotations/test/')

split_dataset()
