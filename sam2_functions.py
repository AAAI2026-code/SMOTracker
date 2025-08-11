import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ffmpeg
from ultralytics import YOLO
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.float).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def create_video_with_ffmpeg(frames_dir="frames",video_name=""):
    """

    :param frames_dir: frames dir for generating video
    :param video_name: name of video
    :return:
    """
    if video_name=="":
        os.system(
            f"ffmpeg -y -framerate 30 -start_number 0 -i {frames_dir}/frame_%d.jpg -c:v ffv1 -pix_fmt yuv420p buffer/video_output/output.avi"
        )
    else:
        os.system(
            f"ffmpeg -y -framerate 30 -start_number 0 -i {frames_dir}/frame_%d.jpg -c:v ffv1 -pix_fmt yuv420p buffer/video_output/{video_name}.avi"
        )
    print(f"video created: buffer/video_output/{video_name}.avi")

class Detector:
    def __init__(self,
                 detector_model:str,
                 save_tag=False,
                 show_tag=False):
        """
        Initialize the Detector class.
        :param detector_model: currently for yolo model path
        :param save_tag: to save the detection result or not
        :param show_tag: show the detection result or not
        """
        self.detector_model=YOLO(detector_model)
        self.save_tag=save_tag
        self.show_tag=show_tag

    def inference(self,
                  media_path,
                  object_classes:list=[0],
                  img_size:int=1024,
                  conf:float=0.35):
        results = self.detector_model.predict(media_path,
                                             imgsz=img_size,
                                             classes=object_classes,
                                             conf=conf,
                                             save=self.save_tag,
                                             show=self.show_tag
                                            )
        return results

def add_new_prompt(inference_state,predictor,frame_id:int,obj_id:int,object_prompt:[]):
    ann_frame_idx = frame_id  # the frame index we interact with
    ann_obj_id = obj_id  # give a unique id to each object we interact with (it can be any integers)
    box = np.array(object_prompt, dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    # prompts[ann_obj_id] = box
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
        clear_old_points=True
    )
    return out_obj_ids,out_mask_logits

def initialize_first_frame_prompt(inference_state,predictor):
    """
    for the first frame, when the detected objects number is 0, will initialize an empty object
    :param inference_state:
    :param predictor:
    :return:
    """
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
    points= np.array([[0,0]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([0], np.int32)
    # prompts[ann_obj_id] = points, labels
    # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels
    )
    return out_obj_ids,out_mask_logits

def clear_dir_content(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def default_ocsort_parser(parser,predictor_dict:dict):
    """
    Initialize the ocsort tracker and SAM2 predictor information
    :param parser:
    :param predictor_dict:
    :return:
    """

    parser.add_argument("--dataset", type=str, default="drone_sam")
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
    parser.add_argument("--TOPIC", type=bool,default=True)

    args = parser.parse_args()

    ocsort_dict=dict(
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
        predictor_dict=predictor_dict,
    )
    return args,ocsort_dict

def mask_to_bbox(mask):
    """
    Transform mask to the bbox position, in form of xyxy
    :param mask:
    :return:
    """
    id_mask = mask.squeeze()
    non_zero_indices = np.argwhere(id_mask)
    bbox=[0,0,0,0]
    if non_zero_indices.size > 0:
        y_min, x_min = non_zero_indices.min(axis=0).tolist()
        y_max, x_max = non_zero_indices.max(axis=0).tolist()
        bbox = [x_min, y_min, x_max, y_max]
    else:
        print("Input mask is without bbox.")
    return  bbox

def record_prediction(mask_memory:dict,file_name:str,h:int,w:int,dir_name:str=""):
    """
    Store the prediction of the model into .txt
    :param mask_memory: in form of mask_memory[fram_id][obj_id]
    :param file_name: to store file name
    :param h: height of the original image
    :param w: width of the original image
    :param dir_name: dir name of the dataset
    :return:
    """
    file_path=os.path.join("prediction_record",dir_name,file_name+".txt")
    file_dir=os.path.join("prediction_record",dir_name)
    os.makedirs(file_dir, exist_ok=True)
    if os.path.exists(file_path):
        os.remove(file_path)

    for frame_id in sorted(mask_memory.keys()):
        for idx,obj_id in enumerate(dict(sorted(mask_memory[frame_id].items()))):
            mask=mask_memory[frame_id][obj_id]
            bbox=mask_to_bbox(mask)
            ab_bbox=[int(b) for b in bbox]
            with open(file_path,"a",encoding="utf-8") as f:
                if not (frame_id==len(mask_memory.keys())-1 and idx==len(mask_memory[frame_id].keys())):
                    f.write(f"{frame_id},{obj_id},{ab_bbox[0]:},{ab_bbox[1]},{ab_bbox[2]},{ab_bbox[3]},1,0,1\n")
                else:
                    f.write(f"{frame_id},{obj_id},{ab_bbox[0]},{ab_bbox[1]},{ab_bbox[2]},{ab_bbox[3]},1,0,1")

def show_mask(mask, ax, obj_id=None, random_color=False,draw_bbox=True,draw_mask=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    # draw mask to the plot
    if draw_mask:
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    # draw bbox to the plot
    if draw_bbox:
        pos = np.argwhere(mask)
        if pos.size > 0:
            id_mask = mask.squeeze()
            non_zero_indices = np.argwhere(id_mask)

            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            w,h=x_max-x_min,y_max-y_min
            # set a rectangular
            rect = patches.Rectangle((x_min, y_min), w, h,
                                     linewidth=2, edgecolor=color[:3], facecolor='none')
            ax.add_patch(rect)
            label_bg_color=(*color[:3],0.8)
            label = f"{obj_id}"
            ax.text(
                x_min, y_min, label,
                fontsize=8,
                color='white',
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(
                    facecolor=label_bg_color,
                    edgecolor='none',
                    alpha=1.0,
                    boxstyle='round,pad=0.3'
                )
            )
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

def extract_frames(video_path, output_dir, fps=None):
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%d.jpg")
    stream = ffmpeg.input(video_path)
    if fps is not None:
        stream = stream.filter('fps', fps=fps)

    stream = stream.output(
        output_template,
        qscale=2,  # 质量参数
        format='image2'  # 输出为图片序列
    )

    # 执行
    ffmpeg.run(stream)

if __name__ == '__main__':
    create_video_with_ffmpeg()