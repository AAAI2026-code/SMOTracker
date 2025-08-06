import os.path
import time
import warnings

import numpy as np
import torch
import cv2
import os

from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task



from external.grounding_dino import video2images as v2i

from PIL import Image
from torch import nn

from external.grounding_dino.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from external.grounding_dino.tools import clear_dir_files,print_gpu_memory_load
from torchvision import transforms

class PostModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.detection_item: list[str] = ["bird","drone","kite","fly","plane"]
        self.detection_item: list[str] = ["drone"]
        self.text_prompt: str = "drone . "
        self.box_threshold: float = 0.3
        self.text_threshold: float = 0.3

    def forward(self, batch):
        """
        Returns Nx5, (x1, y1, x2, y2, conf)
        """

        if isinstance(batch, torch.Tensor):
            _, h, w = batch[0].shape
            boxes, logits, phrases = predict(
                model=self.model,
                image=batch[0],
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
        elif isinstance(batch, str):
            if batch == "":
                raise ValueError(f"batch_path should be with correct content.")
            h, w, _ = cv2.imread(batch).shape
            token = "66049ac32f5235eff5df1590c4b35424"
            config = Config(token)
            client = Client(config)
            image_url = client.upload_file(batch)
            task = V2Task(
                api_path="/v2/task/dinox/detection",
                api_body={
                "model": "DINO-X-1.0",
                "image": image_url,
                "prompt": {
                    "type": "text",
                    "text": self.text_prompt
                },
                "targets": ["bbox"],
                "bbox_threshold": self.box_threshold,
                "iou_threshold": 0.8
            })
            client.run_task(task)
            predictions = task.result["objects"]
            # if not predictions:
            #     raise ValueError(f"The return results are empty")

            boxes = []
            logits = []
            phrases = []
            for idx, obj in enumerate(predictions):
                boxes.append(obj["bbox"])
                logits.append(obj["score"])
                cls_name = obj["category"].lower().strip()
                phrases.append(cls_name)

        # Design the output form.

        outputs = []

        for index, objs in enumerate(phrases):
            objs_list = objs.split(" ")
            obj = objs_list[0]

            if obj in self.detection_item:

                output_result = torch.zeros(5)
                output_result[-1] = logits[index]
                if isinstance(batch,torch.Tensor):
                    x=boxes[index][0]*w
                    y=boxes[index][1]*h
                    w_b=boxes[index][2]*w
                    h_b=boxes[index][3]*h
                elif isinstance(batch,str):
                    x=torch.tensor(boxes[index][0])
                    y=torch.tensor(boxes[index][1])
                    w_b=torch.tensor(boxes[index][2])
                    h_b=torch.tensor(boxes[index][3])
                output_result[0] = torch.round(x - w_b / 2)
                output_result[1] = torch.round(y - h_b / 2)
                output_result[2] = torch.round(x + w_b / 2)
                output_result[3] = torch.round(y + h_b / 2)
                outputs.append(output_result)
        if len(outputs)==0:
            return None
        else:
            return torch.stack(outputs)

def get_model(path):
    model = grounding_dino_load(path)
    # model = model.half()
    model = PostModel(model)
    model.cuda()
    model.eval()
    return model

def grounding_dino_load(path="")->nn.Module:
    model=None
    if path!="":
        model=load_model("external/grounding_dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                   "external/grounding_dino/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    else:
        model=load_model(path)
    return model

if __name__ == '__main__':
    file_name = "bird_drone"  # "MVI_0520_291_379"

    # model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    #                    "GroundingDINO/weights/groundingdino_swint_ogc.pth")
    data_root = "/home/king/PycharmProjects/TOPICTrack_main/buffer"
    annotation_path=os.path.join(data_root,"annotations",file_name,"00001.txt")
    annotation_root=os.path.join(data_root,"annotations",file_name)
    video_root=os.path.join(data_root,"videos",f"{file_name}.mp4")
    images_root = os.path.join(data_root,"video_images","frame_buffer")
    video_output_path = os.path.join(data_root,"video_output",f"{file_name}_drt.mp4")
    results_root = data_root
    frame_step=5
    # Set the text which want to search and show
    text_prompt = "bee ."
    detection_item = ["bee"]
    # Set thresholds for selection and text display
    box_threshold = 0.3
    text_threshold = 0.
    # Switcher for each module
    switcher={
        "split":False,
        "clear":False,
        "dino":True
    }

    # Start execution
    if switcher["clear"]:
        clear_dir_files(images_root)

    start_time = time.time()
    if switcher["split"]:
        video_resized_path = v2i.run_ffmpeg(video_root=video_root,
                                            images_root=images_root,
                                            frames_num=-1,
                                            frame_gaps=frame_step)
    #
    # Read single image as the test
    single_path="/home/king/PycharmProjects/TOPICTrack_main/data/BEE24/train/BEE24-01/img1/000006.jpg"
    print(os.path.isfile(single_path),os.getcwd())
    single_image=Image.open(single_path)
    transform=transforms.ToTensor()
    tensor_image=transform(single_image)
    # print("Video resized",video_resized_path)
    print_gpu_memory_load()
    if switcher["dino"]:
        results=GDINO_run(
                single_image=tensor_image,
                tag={"single":0},
                detection_item=detection_item,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
    # torch.cuda.empty_cache()
    # # print(video_resized_path)
    # print_gpu_memory_load()
    # # Object track with samurai
    #
    # end_time=time.time()
    # print(f"Spend {end_time-start_time}s for this inference")
    print(results)