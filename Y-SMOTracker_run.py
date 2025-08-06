import os

import torch.cuda
from sam2_functions import *
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import load_video_frames
from trackers import ocsort_embedding as tracker_module
from trackers.ocsort_embedding.sam2_association import *
import dataset
import time
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def yolo_sma2_run(video_dir,video_name,buffer_name):
        start_time=time.time()
        # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
        video_height,video_width=0,0
        # Initialize yolo detector
        yolo_model_name="yolov10x_drone.pt"
        yolo_model_path=os.path.join("weights",yolo_model_name)
        yolo_detector=Detector(yolo_model_path)
        # initialize frame buffer path
        frame_buffer_dir=os.path.join("buffer",f"{buffer_name}_frame_buffer")
        frame_output_dir=os.path.join("buffer","video_images",f"{buffer_name}_frames")
        clear_dir_content(frame_buffer_dir)
        clear_dir_content(frame_output_dir)
        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        # initialize sam2 model
        sam2_checkpoint = "weights/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        frame_buffer_list=[]
        frame_buffer_size=9 # in sam2 the usual size for memory is 8, 8*N as base
        video_segments = {}
        obj_count=0
        frame_bias=0
        initial_empty_tag=False
        transfer=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.491, 0.461, 0.406),std=(0.240, 0.220, 0.208)),
             ]
        )
        # execute the inference
        tracker=""
        store_frame_path = ""
        for idx,frame in enumerate(frame_names):
            frame_path = os.path.join(video_dir, frame)
            # store the first frame form last section
            if len(frame_buffer_list)==frame_buffer_size-1:
                store_frame_path=frame_path
            frame_figure = cv2.imread(frame_path)
            if idx==0:
                video_height,video_width,_=frame_figure.shape
            img_tensor=transfer(frame_figure)
            img_numpy=np.array(Image.open(frame_path))
            # track the last frame
            if len(frame_buffer_list)==frame_buffer_size or idx==len(frame_names)-1:
                if idx==len(frame_names)-1:
                    shutil.copy2(frame_path, frame_buffer_dir)
                    frame_buffer_list.append(os.path.join(frame_buffer_dir, frame))
                # initialize/set the inference_state
                if idx==frame_buffer_size:# 1st section
                    frame_bias=int(os.path.splitext(os.path.basename(frame_buffer_list[0]))[0])
                    inference_state = predictor.init_state(video_path=frame_buffer_dir)
                    # initialize association model
                    sam2_dict = {"predictor": predictor, "segments": video_segments, "infer_state": inference_state,
                                 "buffer_size": frame_buffer_size}
                    args, oc_sort_args = default_ocsort_parser(tracker_module.args.make_parser(), sam2_dict)
                    tracker = tracker_module.sam2_association.OCSort(**oc_sort_args)
                # except 1st section
                else:
                    inference_state = predictor.init_state(video_path=frame_buffer_dir)
                    to_update_frame=max(video_segments.keys())
                    # add segmentation memory to initialize new inference state.
                    for obj_id in video_segments[to_update_frame].keys():
                        to_update_mask=video_segments[to_update_frame][obj_id]
                        to_update_bbox=mask_to_bbox(to_update_mask)
                        if sum(to_update_bbox)==0:
                            print(f"{obj_id} in frame {to_update_frame} is without available bbox and to next objection.")
                        else:
                            _,_=add_new_prompt(
                                                inference_state=inference_state,
                                                predictor=predictor,
                                                frame_id=0,
                                                obj_id=obj_id,
                                                object_prompt=to_update_bbox
                                                )
                    tracker.inference_state = inference_state
                # Start tracker update.
                prompts = {} # hold all the clicks we add for visualization
                # as the object id for sam2
                new_boxs=[]
                f_path=frame_buffer_list[0]
                # set section start, final buffer input all directly, others in buffer size
                f_idx_start = int(frame_buffer_list[0].split("/")[-1].split(".")[0])
                f_idx_end = int(frame_buffer_list[-1].split("/")[-1].split(".")[0])
                # execute yolo detection
                result = yolo_detector.inference(f_path,img_size=1280)
                boxs = result[0].boxes.xyxy.cpu()  # transform data from gpu to cpu in form of (x1,y1,x2,y2)
                b_conf=result[0].boxes.conf.cpu() # get the confidence of each box
                boxs_conf=torch.cat((boxs,b_conf.view(-1,1)),dim=1)
                det_objs_count = len(boxs)
                # detect in the frame,first buffer only offers initial frame detection, no match.
                # other buffers offer first frame detection and do matching with the last frame results from last sam2 tracking frame.
                # if not exists, initialize the first section, other sections just pass
                if det_objs_count == 0:
                    # Must give first prompt when using sam2, use (0,0) point as the
                    # initialized prompt and use negative label to avoid mismatching.
                    if idx == frame_buffer_size:
                        initialize_first_frame_prompt(inference_state, predictor)
                        initial_empty_tag=True
                        obj_count += 1
                    else:
                        initial_empty_tag=False
                # add new boxs if exists
                else:
                    initial_empty_tag = False
                    # first section fill the detected boxs and id number
                    if idx == frame_buffer_size:
                        for b in boxs:
                            b=b.tolist()
                            out_frame_idx,out_obj_ids = add_new_prompt(
                                                                    inference_state=inference_state,
                                                                    predictor=predictor,
                                                                    frame_id=0,
                                                                    obj_id=obj_count,
                                                                    object_prompt=b
                                                                    )
                            obj_count+=1
                    # non-first section do the match
                    else:
                        tag = f"frame_id:{max(frame_buffer_list)}"
                        _ = tracker.update(boxs_conf, img_tensor, img_numpy, tag, args.AARM, args.TOPIC,idx-1)
                # get video_segments from annotate images when there are objects

                if len(inference_state["obj_ids"])>0 and not initial_empty_tag:
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                        video_segments[f_idx_start+out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                else:
                    for out_frame_idx in range(f_idx_start, f_idx_end+1, 1):
                        video_segments[out_frame_idx] = {
                        }
                # render the segmentation results every few frames
                vis_frame_stride = 1
                plt.close("all")
                # get out_frame_idx according to the buffer size
                # limit the largest to
                for out_frame_idx in range(f_idx_start, f_idx_end + 1, vis_frame_stride):
                    plt.figure(figsize=(12.8, 7.2))
                    plt.title(f"frame {out_frame_idx}")
                    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx-frame_bias])))
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                    # save frame according to real frame for video generation
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.margins(0, 0)
                    plt.savefig(os.path.join(frame_output_dir, f"frame_{out_frame_idx}.jpg"))
                    plt.close()  # clear the memory
                # initialize for next section of video propagate
                clear_dir_content(frame_buffer_dir)
                frame_buffer_list=[]
                if idx==len(frame_names)-1:
                    pass
                else:
                    shutil.copy2(store_frame_path, frame_buffer_dir)
                    frame_buffer_list.append(store_frame_path)
                    #
                    shutil.copy2(frame_path,frame_buffer_dir)
                    frame_buffer_list.append(os.path.join(frame_buffer_dir,frame))
            else:
                shutil.copy2(frame_path, frame_buffer_dir)
                frame_buffer_list.append(os.path.join(frame_buffer_dir, frame))
        create_video_with_ffmpeg(frame_output_dir,video_name)
        end_time=time.time()
        execution_time=end_time-start_time
        print(f"spend {execution_time} s")
        record_prediction(video_segments,video_name,video_height,video_width,buffer_name)

def process_run(data_dir):
    root_dir="data"
    data_dir_path = os.path.join(root_dir, data_dir, "val")
    for video_dir in os.listdir(data_dir_path):
        video_path = os.path.join(data_dir_path, video_dir, "img1")
        yolo_sma2_run(video_path, video_dir, data_dir)



if __name__ == '__main__':
    torch.cuda.set_device("gpu")
    yolo_sma2_run("data/bird_drone/images/0","bird_drone_test","bird_drone_test")
    # yolo_sma2_run("data/multiple_objects","multiple_objects","multiple_objects")
    # yolo_sma2_run("data/check_data", "check")

    # dir_list=[
    #    "MOT17_Anti-UAV(DUT)" ,
    #    "MOT17_AntiUAV410",
    #    "MOT17_AntiUAV_infrared",
    #    "MOT17_AntiUAV_visible"
    # ]
    # for data_dir in dir_list:
    #     process_run(data_dir)


