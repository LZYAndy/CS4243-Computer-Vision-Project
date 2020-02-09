import cv2
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import os


# Grabs the ground truth training templates from the original dataset

# Read in list of training sets
f = open("datasets/ImageSets/train.txt", "r")
waldo_counter = 0  # there can be more than 1 waldo in a picture
wenda_counter = 0
wizard_counter = 0

for x in f:
    file_name = x.strip()
    print('Reading image ' + file_name)
    img_rgb = cv2.imread('datasets/JPEGImages/' + file_name + '.jpg')
    tree = ET.parse('datasets/Annotations/' + file_name + '.xml')
    root = tree.getroot()

    # make directory
    try:
        dir_name = 'templates/' + file_name
        os.makedirs(dir_name)
    except FileExistsError:
        pass

    for elem in root.findall('object'):
        if elem.find('name').text == 'waldo':
            for bbox in elem.findall('bndbox'):
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                template = img_rgb[ymin:ymax, xmin:xmax]

                write_loc = 'templates/waldo/o_' + str(waldo_counter) + '.jpg'
                cv2.imwrite(write_loc, template)
                waldo_counter += 1
                #print(xmin, xmax, ymin, ymax)

        elif elem.find('name').text == 'wenda':
            for bbox in elem.findall('bndbox'):
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                template = img_rgb[ymin:ymax, xmin:xmax]

                write_loc = 'templates/wenda/o_' + str(wenda_counter) + '.jpg'
                cv2.imwrite(write_loc, template)
                wenda_counter += 1
                #print(xmin, xmax, ymin, ymax)

        elif elem.find('name').text == 'wizard':
            for bbox in elem.findall('bndbox'):
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                template = img_rgb[ymin:ymax, xmin:xmax]

                write_loc = 'templates/wizard/o_' + str(wizard_counter) + '.jpg'
                cv2.imwrite(write_loc, template)
                wizard_counter += 1
    #break
