import cv2
import os
import json

root_dir=os.path.join("data","MVA")
annotations_dir_path=os.path.join(root_dir,"annotations")
annotations_file_path="split_train_mini_coco_frameid"+".json"
image_dir_path=os.path.join(root_dir,"images")
image_file_path="train"

annotations_path=os.path.join(annotations_dir_path,annotations_file_path)
image_dir_path=os.path.join(image_dir_path,image_file_path)
output_dir="buffer"
save_dir= "videos"
print(os.path.exists(annotations_path))

with open(annotations_path, "r") as f:
    content = json.load(f)

images=content["images"]
annotations=content["annotations"]


output_video=None
output=None
frame_width=1920
frame_height=1080

for image in images:
    if image["frame_id"]==0:
        output_video = os.path.join(save_dir, f'{image["video_id"]}.mp4')
        output = cv2.VideoWriter(output_video,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 25,
                                 (frame_width, frame_height))

    image_id=image["id"]
    anno= [ann for ann in annotations if ann["id"] == image_id]
    bboxes=[ann["bbox"] for ann in anno ]
    image_path=os.path.join(image_dir_path,image["video_id"],str(int(image["frame_id"]-1)))
    cv2.imread(os.path.join(image_path, str(image["frame_id"]).rjust(6, '0') + '.jpg'))





