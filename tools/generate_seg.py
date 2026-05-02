import json
import numpy as np
import cv2
import os
import argparse

TRAIN_SET = ['label_data_0114.json']
VAL_SET = ['label_data_0115.json']
TRAIN_VAL_SET = TRAIN_SET + VAL_SET
TEST_SET = ['label_data_0116.json']

def gen_label_for_json(args, image_set):
    H, W = 540, 360
    SEG_WIDTH = 30
    save_dir = args.savedir

    os.makedirs(os.path.join(args.root, args.savedir, "list"), exist_ok=True)
    list_f = open(
        os.path.join(args.root, args.savedir, "list",
                     "{}_gt.txt".format(image_set)), "w")

    json_path = os.path.join(args.root, args.savedir,
                             "{}.json".format(image_set))
    with open(json_path) as f:
        for line in f:
            label = json.loads(line)
            img_path = label['raw_file']

            img_path = img_path.replace("\\", "/")
            print(f"Processed img_path: {img_path}")

            img_dir, img_name = os.path.split(img_path)
            print(f"img_dir: {img_dir}, img_name: {img_name}")

            target_dir = os.path.join(args.root, args.savedir, img_dir.split('/')[-1])
            os.makedirs(target_dir, exist_ok=True)

            seg_file = os.path.join(target_dir, img_name.rsplit('.', 1)[0] + ".png")

            seg_img = np.zeros((H, W, 3))
            list_str = []
            lanes = label['lanes']
            h_samples = label['h_samples']

            for i in range(len(lanes)):
                coords = [(x, y) for x, y in zip(lanes[i], h_samples) if x >= 0]
                if len(coords) < 4:
                    list_str.append('0')
                    continue
                for j in range(len(coords) - 1):
                    cv2.line(seg_img, coords[j], coords[j + 1],
                             (i + 1, i + 1, i + 1), SEG_WIDTH // 2)
                list_str.append('1')

            os.makedirs(os.path.dirname(seg_file), exist_ok=True)

            cv2.imwrite(seg_file, seg_img)

            rel_path = os.path.join(args.savedir, img_dir.split('/')[-1], img_name.rsplit('.', 1)[0] + ".png")
            abs_img_path = os.path.join(args.root, args.savedir, img_dir.split('/')[-1], img_name)
            abs_seg_path = os.path.abspath(rel_path)

            list_str.insert(0, abs_seg_path)
            list_str.insert(0, abs_img_path)
            list_str = " ".join(list_str) + "\n"
            list_f.write(list_str)

def generate_json_file(save_dir, json_file, image_set):
    with open(os.path.join(save_dir, json_file), "w") as outfile:
        for json_name in (image_set):
            with open(os.path.join(args.root, json_name)) as infile:
                for line in infile:
                    outfile.write(line)


def generate_label(args):
    save_dir = os.path.join(args.root, args.savedir)
    os.makedirs(save_dir, exist_ok=True)
    generate_json_file(save_dir, "train_val.json", TRAIN_VAL_SET)
    generate_json_file(save_dir, "test.json", TEST_SET)

    print("generating train_val set...")
    gen_label_for_json(args, 'train_val')
    print("generating test set...")
    gen_label_for_json(args, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        required=False,
                        default=r'D:\MFPLNet2024\MFPLNet2024\mfplnet\data\cpld_02',
                        help='The root of the Tusimple dataset')
    parser.add_argument('--savedir',
                        type=str,
                        default='seg_label',
                        help='The root of the Tusimple dataset')
    args = parser.parse_args()

    generate_label(args)