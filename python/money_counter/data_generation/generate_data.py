import cv2
import argparse
import random
import numpy as np
import xml.etree.cElementTree as ET

background = cv2.imread('assets/background.jpg')
hands = []
for i in range(1, 5):
    hands.append(cv2.imread(f'assets/hand{i}.jpg'))

coin5 = cv2.imread('assets/5.png')
coin10 = cv2.imread('assets/10.png')
coin20 = cv2.imread('assets/20.png')
coin50 = cv2.imread('assets/50.png')
coin1 = cv2.imread('assets/1.png')
coin2 = cv2.imread('assets/2.png')
coins = {
    5: coin5, 
    10: coin10,
    20: coin20,
    50: coin50,
    1: coin1,
    2: coin2
}

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--num_train", type=int, default=1, help="The number of training images to generate (default: 1)")
parser.add_argument("--num_val", type=int, default=1, help="The number of validation images to generate (default: 1)")
parser.add_argument("--num_test", type=int, default=1, help="The number of test images to generate (default: 1)")


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def generate_image(background: np.ndarray):
    coordinates = {
        5: [],
        10: [],
        20: [],
        50: [],
        1: [],
        2: []
    }
    working_background = np.copy(background)
    for coin_num, coin in coins.items():
        num_to_generate = random.randint(0, 10)
        for _ in range(num_to_generate):
            working_copy = np.copy(coin)

            # Get coin mask
            colour_mask = np.array([0, 255, 0]) # Green
            coin_mask = (cv2.inRange(working_copy, colour_mask, colour_mask) == 0).astype(int)
            coin_mask[coin_mask == 0] = -1

            # Rotate
            rotation_angle = random.randint(0, 360)
            working_copy = rotate_image(working_copy, rotation_angle)
            coin_mask = rotate_image(coin_mask)

            # Place onto canvas
            top = random.randint(0, working_background.shape[0] - working_copy.shape[0] - 1)
            left = random.randint(0, working_background.shape[1] - working_copy.shape[1] - 1)
            canvas = np.ones_like(background) * -1
            canvas[top:top + working_copy.shape[0], left:left + working_copy.shape[1]] = coin_mask

            # Place onto image
            working_background[canvas != -1] = working_copy[coin_mask != -1]

            # Store metadata
            coordinates[coin_num].append(top, left, top + working_copy.shape[0], left + working_copy.shape[1])
    
    return working_background, coordinates


def generate_annotation_file(coordinates, filename: str, width: int, height: int):
    database_name = 'coins'
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = f"{filename}.jpg"
    ET.SubElement(root, "folder").text = database_name

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = database_name
    ET.SubElement(source, "annotation").text = 'custom'
    ET.SubElement(source, "image").text = 'custom'

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = width
    ET.SubElement(size, "height").text = height
    ET.SubElement(size, "depth").text = 3
    ET.SubElement(size, "segmented").text = 0

    for coin_type, anno_list in coordinates:
        for annotation in anno_list:
            det_obj = ET.SubElement(root, "object")
            ET.SubElement(det_obj, "name").text = str(coin_type)
            ET.SubElement(det_obj, "pose").text = 'unspecified'
            ET.SubElement(det_obj, "truncated").text = '0'
            ET.SubElement(det_obj, "difficult").text = '0'

            bbox = ET.SubElement(det_obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = annotation[1]
            ET.SubElement(bbox, "ymin").text = annotation[0]
            ET.SubElement(bbox, "xmax").text = annotation[3]
            ET.SubElement(bbox, "ymax").text = annotation[2]

    tree = ET.ElementTree(root)
    tree.write(f"{filename}.xml")



args = parser.parse_known_args()[0]

for _ in range(args.num_train):
    generated_image, coordinates = generate_image(background)
    generate_annotation_file(coordinates, 'test_data', generated_image.shape[1], generated_image.shape[0])
    
