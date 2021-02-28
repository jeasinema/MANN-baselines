# -*- code: utf-8 -*-


import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm


def process_attr_one_comp(files):
    for file in tqdm(files):
        xml_tree = ET.parse(file)
        xml_tree_root = xml_tree.getroot()
        xml_panels = xml_tree_root[0]
        # set the second dim to be the same as max number of obejects in it
        # set to 4 if 2x2 grid, 9 if 3x3 grid, and 1 if center_single (possibly this value)
        # if one slot has no object in it, default values for type, size, and color are set to max_idx
        exist = np.zeros((16, 9))
        type = np.zeros((16, 9)) + 4
        size = np.zeros((16, 9)) + 5
        color = np.zeros((16, 9)) + 9
        for i in range(16):
            panel = xml_panels[i][0][0][0]
            all_position = eval(panel.attrib["Position"])
            for entity in panel:
                pos = eval(entity.attrib["bbox"])
                pos_index = all_position.index(pos)
                exist[i, pos_index] = 1
                type_index = int(entity.attrib["Type"])
                type[i, pos_index] = type_index - 1
                size_index = int(entity.attrib["Size"])
                size[i, pos_index] = size_index
                color_index = int(entity.attrib["Color"])
                color[i, pos_index] = color_index
        new_file = file.replace(".xml", "_attr.npz")
        np.savez(new_file, exist=exist,
                        type=type,
                        size=size,
                        color=color)

def process_attr_two_comp(files):
    for file in tqdm(files):
        xml_tree = ET.parse(file)
        xml_tree_root = xml_tree.getroot()
        xml_panels = xml_tree_root[0]
        # this is the case for in_distribute_four_out_center_single, so maximum number of objects = 5
        # for up_down / left_right / in_center_single_out_center_single the number is 2
        exist = np.zeros((16, 5))
        type = np.zeros((16, 5)) + 4
        size = np.zeros((16, 5)) + 5
        color = np.zeros((16, 5)) + 9
        for i in range(16):
            # for in_out xml_panels[i][0][1][0] is in, xml_panels[i][0][0][0] is out
            # for left_right xml_panels[i][0][1][0] is right, xml_panels[i][0][0][0] is left
            # for up_down xml_panels[i][0][1][0] is down, xml_panels[i][0][0][0] is up
            in_panels = xml_panels[i][0][1][0]
            all_position = eval(in_panels.attrib["Position"])
            for entity in in_panels:
                pos = eval(entity.attrib["bbox"])
                pos_index = all_position.index(pos)
                exist[i, pos_index] = 1
                type_index = int(entity.attrib["Type"])
                type[i, pos_index] = type_index - 1
                size_index = int(entity.attrib["Size"])
                size[i, pos_index] = size_index
                color_index = int(entity.attrib["Color"])
                color[i, pos_index] = color_index
            out_panel = xml_panels[i][0][0][0]
            for entity in out_panel:
                pos_index = 4
                exist[i, pos_index] = 1
                type_index = int(entity.attrib["Type"])
                type[i, pos_index] = type_index - 1
                size_index = int(entity.attrib["Size"])
                size[i, pos_index] = size_index
                color_index = int(entity.attrib["Color"])
                color[i, pos_index] = color_index
        new_file = file.replace(".xml", "_attr.npz")
        np.savez(new_file, exist=exist,
                        type=type,
                        size=size,
                        color=color)

def main():
    path = "/home/robot/workspace/eb_lang_learner/dataset/raven"
    configs = [
        ("center_single", 1),
        ("distribute_four", 1),
        ("distribute_nine", 1),
        ("left_center_single_right_center_single", 2),
        ("up_center_single_down_center_single", 2),
        ("in_center_single_out_center_single", 2),
        ("in_distribute_four_out_center_single", 2),
    ]
    for config, num_comp in configs:
        print('Config: {}'.format(config))
        files = glob.glob(os.path.join(path, config, "*.xml"))
        if num_comp == 1:
            process_attr_one_comp(files)
        else:
            process_attr_two_comp(files)

if __name__ == '__main__':
    main()