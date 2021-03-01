# -*- code: utf-8 -*-


import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm


def process_attr_one_comp(files, dim=1):
    for file in tqdm(files):
        xml_tree = ET.parse(file)
        xml_tree_root = xml_tree.getroot()
        xml_panels = xml_tree_root[0]
        # set the second dim to be the same as max number of obejects in it
        # set to 4 if 2x2 grid, 9 if 3x3 grid, and 1 if center_single (possibly this value)
        # if one slot has no object in it, default values for type, size, and color are set to max_idx
        exist = np.zeros((16, dim))
        type = np.zeros((16, dim)) + 4
        size = np.zeros((16, dim)) + 5
        color = np.zeros((16, dim)) + 9
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

def process_attr_two_comp(files, dim=2):
    for file in tqdm(files):
        xml_tree = ET.parse(file)
        xml_tree_root = xml_tree.getroot()
        xml_panels = xml_tree_root[0]
        # this is the case for in_distribute_four_out_center_single, so maximum number of objects = 5
        # for up_down / left_right / in_center_single_out_center_single the number is 2
        # if one slot has no object in it, default values for type, size, and color are set to max_idx
        exist = np.zeros((16, dim))
        type = np.zeros((16, dim)) + 4
        size = np.zeros((16, dim)) + 5
        color = np.zeros((16, dim)) + 9
        for i in range(16):
            # for in_out xml_panels[i][0][1][0] is in, xml_panels[i][0][0][0] is out
            # for left_right xml_panels[i][0][1][0] is right, xml_panels[i][0][0][0] is left
            # for up_down xml_panels[i][0][1][0] is down, xml_panels[i][0][0][0] is up
            in_panels = xml_panels[i][0][1][0]
            all_position = eval(in_panels.attrib["Position"])
            for entity in in_panels:
                if dim == 2:
                    pos_index = 0
                # in_distribute_four_out_center_single
                else:
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
                if dim == 2:
                    pos_index = 1
                # in_distribute_four_out_center_single
                else:
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
        ("center_single", 1, 1),
        ("distribute_four", 1, 4),
        ("distribute_nine", 1, 9),
        ("left_center_single_right_center_single", 2, 2),
        ("up_center_single_down_center_single", 2, 2),
        ("in_center_single_out_center_single", 2, 2),
        ("in_distribute_four_out_center_single", 2, 5),
    ]
    for config, num_comp, dim in configs:
        print('Config: {}'.format(config))
        files = glob.glob(os.path.join(path, config, "*.xml"))
        if num_comp == 1:
            process_attr_one_comp(files, dim)
        else:
            process_attr_two_comp(files, dim)

if __name__ == '__main__':
    main()
