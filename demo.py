import sys
import pdb
import os
import shutil
import time

import torch
import cv2
import numpy as np

import dataset
import utils
from external.adaptors import detector
from trackers import ocsort_embedding as tracker_module

def get_main_args():
    parser = tracker_module.args.make_parser()
    parser.add_argument("--dataset", type=str, default="dance")
    parser.add_argument("--result_folder", type=str,
                        default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--min_box_area", type=float,
                        default=10, help="filter out tiny boxes")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument("--post", type=bool, default=True,
                        help="run post-processing linear interpolation.",)
    parser.add_argument("--w_assoc_emb", type=float,
                        default=0.75, help="Combine weight for emb cost")
    parser.add_argument(
        "--alpha_gate",
        type=float,
        default=0.9,
        help="alpha_gate",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=1,
        help="gate",
    )
    parser.add_argument(
        "--gate2",
        type=float,
        default=0.3,
        help="gate",
    )

    parser.add_argument("--new_kf_off", type=bool, default=True)
    parser.add_argument("--AARM", action="store_true")
    parser.add_argument("--TOPIC", action="store_true")

    args = parser.parse_args()

    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    elif args.dataset == "dance":
        args.result_folder = os.path.join(args.result_folder, "DANCE-val")
    elif args.dataset == "BEE24":
        args.result_folder = os.path.join(args.result_folder, "BEE24-val")
    elif args.dataset == "gmot":
        args.result_folder = os.path.join(args.result_folder, "GMOT-val")
    elif args.dataset == "mav":
        args.result_folder = os.path.join(args.result_folder, "MAV-val")
    elif args.dataset == "bird_drone":
        args.result_folder = os.path.join(args.result_folder, "bird_drone-val")
    if args.test_dataset:
        args.result_folder.replace("-val", "-test")
    return args

def sort_output(txt_path):
    with open(txt_path, 'r') as f:
        list = []
        for line in f:
            list.append(line.strip())

    with open(txt_path, "w") as f:
        for item in sorted(list, key=lambda x: int(str(x).split(',')[0])):
            f.writelines(item)
            f.writelines('\n')
        f.close()

def draw_dataset(video_id, gt_path, dataset, save_dir,folder_name:str="result"):
    exclusion = ["mot17", "mot20", "dance", "BEE24", "gmot"]
    txt_name = gt_path + '/' + video_id + '.txt'
    if dataset in exclusion:
        file_path_img = f'data/{dataset}/test/' + video_id + '/img1'
    else:
        file_path_img = f'data/{dataset}/test/' + video_id

    frame_width = 1920
    frame_height = 1080
    output_video = os.path.join(save_dir, f'{video_id}.mp4')
    output = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(
        *'mp4v'), 25, (frame_width, frame_height))
    save_file_path = os.path.join(save_dir, video_id)
    sort_output(txt_name)

    source_file = open(txt_name)
    records = source_file.readlines()

    i = 0
    img_names = []
    for line in records:
        staff = line.split(',')
        img_name = staff[0]
        img_names.append(img_name)
        if i == 0:
            num = int(float(staff[1])) - 1
        i += 1

    name_dict = {}
    for i in img_names:
        if img_names.count(i):
            name_dict[i] = img_names.count(i)

    l = 0
    for idx in name_dict:
        # print(os.path.join(
        #     file_path_img, str(idx).rjust(6, '0') + '.jpg'))

        if dataset in exclusion:
            img = cv2.imread(os.path.join(
                file_path_img, str(idx).rjust(6, '0') + '.jpg'))
        else:
            img = cv2.imread(os.path.join(
                file_path_img, str(int(idx)-1) + '.jpg'))
        for i in range(name_dict[idx]):

            line = records[l]
            l += 1
            staff = line.split(',')

            id = staff[1]
            frame = staff[0]
            cls = staff[7]
            box = staff[2:6]
            cv2.rectangle(img, (int(float(box[0])), int(float(box[1]))),
                          (int(float(box[0])) + int(float(box[2])),
                           int(float(box[1])) + int(float(box[3]))),
                          (255, 255, 0), 2, cv2.LINE_4)
            cv2.rectangle(img, (int(float(box[0])), int(float(box[1]))),
                          ((int(float(box[0])) + 20),
                           (int(float(box[1])) + 15)),
                          (255, 255, 0), thickness=-1)
            cv2.putText(img, str(int(float(id))), (int(float(box[0])), int(float(box[1]))+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        txt = f'{dataset}  ' + 'frame: ' + str(frame) + '  num: ' + str(i + 1)
        cv2.putText(img, txt, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        img = cv2.resize(img, dsize=(1920, 1080))

        output.write(img)

    output.release()
    source_file.close()

def process_det(det_box):
    min_area = 432
    max_area = 10710

    filtered_boxes_list = det_box.tolist()
    filtered_boxes_filtered = []
    for box in filtered_boxes_list:
        xmin, ymin, xmax, ymax, _ = box
        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        # if 10<=width<=157 and 10<=height<=157:
        #     filtered_boxes_filtered.append(box)
        if area > min_area and area < max_area:
            filtered_boxes_filtered.append(box)

    filtered_boxes_tensor = torch.tensor(filtered_boxes_filtered)
    return filtered_boxes_tensor

def main():
    np.set_printoptions(suppress=True, precision=5)
    args = get_main_args()

    if args.dataset == "mot17":
        if args.test_dataset:
            detector_path = "external/weights/topictrack_mot17.pth.tar"
        else:
            detector_path = "external/weights/topictrack_ablation.pth.tar"
        size = (800, 1440)
    elif args.dataset == "mot20":
        if args.test_dataset:
            detector_path = "external/weights/topictrack_mot20.tar"
            size = (896, 1600)
        else:
            detector_path = "external/weights/topictrack_mot17.pth.tar"
            size = (800, 1440)
    elif args.dataset == "dance":
        detector_path = "external/weights/topictrack_dance.pth.tar"
        size = (800, 1440)
    elif args.dataset == "BEE24":
        detector_path = "external/weights/topictrack_bee24.pth.tar"
        size = (800, 1440)
    elif args.dataset == "gmot":
        detector_path = "external/weights/topictrack_gmot.pth.tar"
        size = (800, 1440)
    elif args.dataset == "MVA":
        detector_path = "external/weights/yolo_x_mva.pth.tar"
        size = (800, 1440)
    elif args.dataset == "bird_drone":
        detector_path = "external/weights/yolo_x_mva.pth.tar"
        size = (800, 1440)
    else:
        raise RuntimeError(
            "Need to update paths for detector for extra datasets.")
    #
    det = detector.Detector("yolox", detector_path, args.dataset)
    # det = detector.Detector("g_dino", detector_path, args.dataset)
    loader = dataset.get_mot_loader(args.dataset, args.test_dataset, size=size)

    oc_sort_args = dict(
        args=args,
        det_thresh=args.track_thresh,
        alpha_gate=args.alpha_gate,
        gate=args.gate,
        gate2=args.gate2,
        iou_threshold=args.iou_thresh,
        asso_func=args.asso,
        delta_t=args.deltat,
        inertia=args.inertia,
        w_association_emb=args.w_assoc_emb,
        new_kf_off=args.new_kf_off,
    )
    print(oc_sort_args)
    tracker = tracker_module.ocsort.OCSort(**oc_sort_args)
    results = {}
    frame_count = 0
    total_time = 0
    for (img, np_img), label, info, idx in loader:

        frame_id = info[2].item()

        video_name = info[4][0].split("/")[0]

        tag = f"{video_name}:{frame_id}"
        # print(tag)
        if video_name not in results:
            results[video_name] = []
        img = img.cuda()

        if frame_id == 1:

            tracker.dump_cache()
            tracker = tracker_module.ocsort.OCSort(**oc_sort_args)

        start_time = time.time()

        img_path=os.path.join("data",args.dataset,"test",info[4][0])
        pred = det(img, tag)

        if 'BEE24' in video_name and pred!=None:
            pred = process_det(pred)

        if pred==None or len(pred)==0:
            continue
        # print(pred, img, np_img[0].numpy(), tag)
        targets = tracker.update(
            pred, img, np_img[0].numpy(), tag, args.AARM, args.TOPIC)
        print(targets)
        tlwhs, ids = utils.filter_targets(
            targets, args.aspect_ratio_thresh, args.min_box_area, args.dataset)

        total_time += time.time() - start_time
        frame_count += 1

        results[video_name].append((frame_id, tlwhs, ids))
    #
    # det.dump_cache()
    # tracker.dump_cache()
    #
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")
    if args.post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        utils.dti(post_folder_data, post_folder_data)
        print(
            f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

    filename = os.listdir(folder)
    for name in filename:
        print('The videos ' + name.split('.')[0] + ' begin!')
        draw_dataset(name.split('.')[0], folder,
                     args.dataset, save_dir="videos/", folder_name=name)

        print('The videos ' + name.split('.')[0] + ' Done!')

def draw(name, pred, i):
    pred = pred.cpu().numpy()
    name = os.path.join("data/mot/train", name)
    img = cv2.imread(name)
    for s in pred:
        p = np.round(s[:4]).astype(np.int32)
        cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 3)
    for s in pred:
        p = np.round(s[:4]).astype(np.int32)
        cv2.putText(
            img,
            str(int(round(s[4], 2) * 100)),
            (p[0] + 20, p[1] + 20),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            thickness=3,
        )
    cv2.imwrite(f"debug/{i}.png", img)

if __name__ == "__main__":
    main()
