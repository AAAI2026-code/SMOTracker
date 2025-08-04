import os.path
import time
import torch
import cv2
import os

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def GDINO_detector( 
        single_image_path,
        detection_item="drone",
        text_prompt="drone . ",
        box_threshold=0.35,
        text_threshold=0.25
        ):
    # Create annotation and results folders
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "GroundingDINO/weights/groundingdino_swint_ogc.pth")
    # Single image detection and annotation
    image_name=single_image_path.split("/")[-1].split(".")[0]
    img_figure=cv2.imread(single_image_path)
    h,w,_=img_figure.shape
    image_source, image = load_image(single_image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    bboxs=[]
    confidence_list=[]
    for i in range(len(boxes)):
        if phrases[i]==detection_item:
            bbox=[boxes[i][0]*w,boxes[i][1]*h,boxes[i][2]*w,boxes[i][3]*h]
            bbox=[bbox[0]-bbox[2]/2,bbox[1]-bbox[3]/2,bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2]
            bbox=[int(v) for v in bbox]
            print(bbox,logits[i],phrases[i])
            bboxs.append(bbox)
            confidence_list.append(logits[i])
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(f'{os.path.join("GroundingDINO","results",image_name)}_anno.jpg', annotated_frame)
    print("Grounding Dino results are",bboxs)
    return [bboxs,confidence_list]

if __name__ == '__main__':
    start_time=time.time()
    img_path="data/MOT17_Anti-UAV(DUT)/val/AntiUAV-video01/img1/000001.jpg"
    #
    text_prompt = "drone . "
    box_threshold = 0.35
    text_threshold = 0.25
    detection_item = "drone"
    GDINO_detector(
            img_path,
            detection_item,
            text_prompt,
            box_threshold,
            text_threshold
        )
    torch.cuda.empty_cache()
    end_time=time.time()
    print(f"spend {end_time-start_time} s")
