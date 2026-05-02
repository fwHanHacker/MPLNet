import cv2
import os
import glob
import json
import numpy as np
import shutil

print("[LOG] start ...")
labelme_dir = r"D:\MFPLNet2024\data002\data\test"

result_dir = r"D:\MFPLNet2024\MFPLNet2024\mfplnet\data\cpld_02"

json_file_name = "label_data_0116.json"
clips_sub_name = os.path.splitext(json_file_name)[0].split("_")[-1]
clips_sub = os.path.join("clips", clips_sub_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
clips_sub_dir = os.path.join(result_dir, clips_sub)
if not os.path.exists(clips_sub_dir):
    os.makedirs(clips_sub_dir)

h_samples = list(range(0, 540, 1))
print("h_sample:", h_samples)

labelme_img = glob.glob(labelme_dir + '/*.jpg')
print("labelme_json:", labelme_img)

json_file_path = os.path.join(result_dir, json_file_name)
with open(json_file_path, 'w') as wr:
    for idx, img_path in enumerate(labelme_img):
        print(idx, img_path)
        img_name_file = os.path.basename(img_path)
        img_name = os.path.splitext(img_name_file)[0]
        json_path = os.path.splitext(img_path)[0] + ".json"
        lanes = []
        dict_img_per = {}
        if os.path.isfile(json_path):
            img = cv2.imread(img_path, -1)
            img_w = img.shape[1]
            img_h = img.shape[0]

            binary_image_h = np.zeros([img_h, img_w], np.uint8)
            for h in h_samples:
                cv2.line(binary_image_h, (0, h), (img_w - 1, h), (255), thickness=1)

            with open(json_path, 'r') as json_obj:
                data = json.load(json_obj)
                for shapes in data['shapes']:
                    binary_image = np.zeros([img_h, img_w], np.uint8)
                    single_lane = []

                    label = shapes['label']
                    points = shapes['points']
                    points = np.array(points, dtype=int)
                    cv2.polylines(binary_image, [points], False, (255), thickness=1)
                    img_and = cv2.bitwise_and(binary_image, binary_image_h)

                    for h in h_samples:
                        start = False
                        temp_w = []
                        for w in range(img_w):
                            if img_and[h, w] >= 1:
                                start = True
                                temp_w.append(w)
                        if start:
                            half = len(temp_w) // 2
                            median = (temp_w[half] + temp_w[~half]) / 2
                            median = int(median)
                            single_lane.append(median)
                        else:
                            single_lane.append(-2)

                    lanes.append(single_lane)
            json_obj.close()
            print("lanes:", lanes)
            raw_file = os.path.join(clips_sub, img_name_file)
            shutil.copy(img_path, clips_sub_dir)
            dict_img_per = {"lanes": lanes, "h_samples": h_samples, "raw_file": raw_file}
            json.dump(dict_img_per, wr)
            wr.write('\n')
        else:
            continue
    wr.close()