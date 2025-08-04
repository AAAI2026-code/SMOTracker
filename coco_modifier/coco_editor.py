import json
import os.path
import shutil
import time
from pathlib import Path
from typing import Dict, List, Union
from coco_tracker import COCOTracker


class COCOEditor:
    def __init__(self, annotation_path: str):
        """
        COCO 标注文件编辑器
        参数:
            annotation_path: COCO 格式标注文件路径 (通常为 .json)
        """
        self.annotation_path = Path(annotation_path)
        self.backup_path = self.annotation_path.with_suffix('.bak.json')
        self.data: Dict = self._load_annotations()

        # 建立快速查找索引
        self.image_id_map = {img['id']: img for img in self.data['images']}
        self.ann_id_map = {ann['id']: ann for ann in self.data['annotations']}
        self.cat_id_map = {cat['id']: cat for cat in self.data['categories']}
        self.cat_name_map = {cat['name']: cat for cat in self.data['categories']}
        # 新增帧序列管理属性
        self.frame_sequence = []  # 按视频顺序排列的image_id列表
        self.frame_id_map = {}  # image_id到frame_id的映射
        # 新增视频管理属性
        self.video_id_map = {}  # video_id 到视频信息的映射
        self.video_frames = {}  # video_id 到对应帧列表的映射
        #
        self.tracker=COCOTracker()

    def _load_annotations(self) -> Dict:
        """加载并验证 COCO 标注文件"""
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"标注文件 {self.annotation_path} 不存在")

        try:
            with open(self.annotation_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("文件格式错误，不是有效的 JSON 文件")

        # 验证必要字段
        required_keys = {'images', 'annotations', 'categories'}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - data.keys()
            raise ValueError(f"缺少必要字段: {missing}")

        return data

    def backup_original(self) -> None:
        """创建原始文件的备份"""
        shutil.copy(self.annotation_path, self.backup_path)
        print(f"已创建备份文件: {self.backup_path}")

    def add_category(self, name: str, supercategory: str = 'object') -> int:
        """
        添加新类别
        参数:
            name: 类别名称 (必须唯一)
            supercategory: 父类别名称
        返回:
            新类别的 ID
        """
        if name in self.cat_name_map:
            raise ValueError(f"类别名称 '{name}' 已存在")

        # 生成新 ID (现有最大 ID +1)
        new_id = max(self.cat_id_map.keys()) + 1 if self.cat_id_map else 1

        new_category = {
            "id": new_id,
            "name": name,
            "supercategory": supercategory
        }

        self.data['categories'].append(new_category)
        self.cat_id_map[new_id] = new_category
        self.cat_name_map[name] = new_category
        return new_id

    def remove_annotations(self, image_ids: List[int] = None, ann_ids: List[int] = None) -> None:
        """
        删除指定标注
        参数:
            image_ids: 要删除的图片 ID 列表 (删除该图片下所有标注)
            ann_ids: 要删除的标注 ID 列表
        """
        if image_ids is None and ann_ids is None:
            raise ValueError("必须指定 image_ids 或 ann_ids")

        # 通过图片 ID 查找标注
        if image_ids:
            anns_to_remove = [
                ann for ann in self.data['annotations']
                if ann['image_id'] in image_ids
            ]
            ann_ids = list({ann['id'] for ann in anns_to_remove})

        # 删除标注
        self.data['annotations'] = [
            ann for ann in self.data['annotations']
            if ann['id'] not in ann_ids
        ]

        # 更新索引
        for ann_id in ann_ids:
            self.ann_id_map.pop(ann_id, None)

    def modify_bbox(self, ann_id: int, new_bbox: List[float]) -> None:
        """
        修改标注框坐标
        参数:
            ann_id: 标注 ID
            new_bbox: 新边界框 [x, y, width, height]
        """
        if ann_id not in self.ann_id_map:
            raise KeyError(f"标注 ID {ann_id} 不存在")

        if len(new_bbox) != 4:
            raise ValueError("边界框必须是 [x, y, width, height] 格式")

        ann = self.ann_id_map[ann_id]
        ann['bbox'] = new_bbox

        # 自动更新面积 (可选)
        ann['area'] = new_bbox[2] * new_bbox[3]

    def save(self, output_path: Union[str, None] = None) -> None:
        """
        保存修改后的标注文件
        参数:
            output_path: 输出路径 (默认覆盖原文件)
        """
        output_path = Path(output_path) if output_path else self.annotation_path

        # 验证数据结构
        self._validate_structure()

        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2)

        print(f"标注文件已保存至: {output_path}")

    def _validate_structure(self) -> None:
        """验证修改后的数据结构"""
        # 检查 ID 唯一性
        image_ids = {img['id'] for img in self.data['images']}
        if len(image_ids) != len(self.data['images']):
            raise ValueError("存在重复的图片 ID")

        ann_ids = {ann['id'] for ann in self.data['annotations']}
        if len(ann_ids) != len(self.data['annotations']):
            raise ValueError("存在重复的标注 ID")

        # 检查外键引用
        for ann in self.data['annotations']:
            if ann['image_id'] not in image_ids:
                raise ValueError(f"标注 {ann['id']} 引用了不存在的图片 ID {ann['image_id']}")
            if ann['category_id'] not in self.cat_id_map:
                raise ValueError(f"标注 {ann['id']} 引用了不存在的类别 ID {ann['category_id']}")
            # 新增frame_id检查
            if hasattr(self, 'video_frames'):
                for video_id, frame_list in self.video_frames.items():
                    frame_ids = [self.frame_id_map[img_id] for img_id in frame_list]
                    if len(frame_ids) != len(set(frame_ids)):
                        raise ValueError(f"视频 {video_id} 的 frame_id 不唯一")
                    if min(frame_ids) != 1:
                        raise ValueError(f"视频 {video_id} 的 frame_id 未从 1 开始")

    def build_frame_sequence(self,
                             sort_key: str = 'file_name',
                             start_image_id: int = None,
                             image_ids: List[int] = None) -> Dict[int, int]:
        """
        构建视频帧序列并分配 frame_id
        参数:
            sort_key: 排序依据字段 ('file_name' 或 'timestamp')
            start_image_id: 指定起始帧的 image_id (默认为序列第一帧)
            image_ids: 指定需要构建帧序列的图片 ID 列表 (默认为所有图片)
        返回:
            frame_id_map: 当前视频的 image_id 到 frame_id 的映射
        """
        if sort_key not in ['file_name', 'timestamp']:
            raise ValueError("仅支持 file_name 或 timestamp 排序")

        # 过滤需要排序的图片
        images_to_sort = [
            img for img in self.data['images']
            if image_ids is None or img['id'] in image_ids
        ]

        # 获取可排序的元组列表 (image_id, 排序键值)
        sortable = []
        for img in images_to_sort:
            if sort_key == 'file_name':
                # 从文件名提取数字部分 (如 "frame_000123.jpg" -> 123)
                num_str = ''.join(filter(str.isdigit, img['file_name']))
                if not num_str:
                    raise ValueError(f"文件名 {img['file_name']} 无数字序号")
                key = int(num_str)
            else:
                key = img.get('timestamp', 0)
            sortable.append((img['id'], key))

        # 按排序键值排序
        sorted_items = sorted(sortable, key=lambda x: x[1])
        frame_sequence = [item[0] for item in sorted_items]

        # 确定起始位置
        start_idx = 0
        if start_image_id is not None:
            try:
                start_idx = frame_sequence.index(start_image_id)
            except ValueError:
                raise ValueError(f"起始图片ID {start_image_id} 不在序列中")

        # 生成当前视频的 frame_id 映射
        frame_id_map = {}
        current_frame_id = 1
        for img_id in frame_sequence[start_idx:]:
            frame_id_map[img_id] = current_frame_id
            self.image_id_map[img_id]['frame_id'] = current_frame_id
            current_frame_id += 1

        return frame_id_map

    def get_adjacent_frames(self, image_id: int, n_prev: int = 5, n_next: int = 5) -> List[int]:
        """
        获取相邻帧的image_id列表
        参数:
            image_id: 当前帧ID
            n_prev: 前向帧数
            n_next: 后向帧数
        返回:
            相邻帧image_id列表 (包含当前帧)
        """
        try:
            idx = self.frame_sequence.index(image_id)
        except ValueError:
            return []

        start = max(0, idx - n_prev)
        end = min(len(self.frame_sequence), idx + n_next + 1)
        return self.frame_sequence[start:end]

    def extract_video_id_from_path(self, file_name: str) -> str:
        """
        从文件路径提取 video_id (文件夹名称)
        参数:
            file_name: 图片文件名 (如 "video1/frame_0001.jpg")
        返回:
            video_id (如 "video1")
        """
        # 提取第一层文件夹名
        parts = file_name.split('/')
        if len(parts) < 2:
            raise ValueError(f"文件路径 {file_name} 未包含文件夹层级")
        return parts[0]

    def assign_video_ids(self) -> None:
        """为所有图片分配 video_id"""
        for img in self.data['images']:
            video_id = self.extract_video_id_from_path(img['file_name'])

            img['video_id'] = video_id


            # 更新视频帧映射
            if video_id not in self.video_frames:
                self.video_frames[video_id] = []
            self.video_frames[video_id].append(img['id'])

    def build_frame_sequence_by_video(self,
                                      sort_key: str = 'file_name',
                                      start_image_id: int = None) -> None:
        """
        按 video_id 分组构建帧序列
        参数:
            sort_key: 排序依据字段 ('file_name' 或 'timestamp')
            start_image_id: 指定起始帧的 image_id (默认为序列第一帧)
        """
        if not self.video_frames:
            self.assign_video_ids()  # 确保已分配 video_id

        # 清空全局 frame_id 映射
        self.frame_id_map = {}

        # 对每个视频单独构建帧序列
        for video_id, frame_list in self.video_frames.items():
            # 获取当前视频的 frame_id 映射
            video_frame_id_map = self.build_frame_sequence(
                sort_key=sort_key,
                start_image_id=start_image_id,
                image_ids=frame_list  # 仅处理当前视频的帧
            )
            # 更新全局 frame_id 映射
            self.frame_id_map.update(video_frame_id_map)

    def get_video_info(self, video_id: str) -> Dict:
        """
        获取指定视频的信息
        返回:
            {
                "frame_count": int,
                "start_frame": int,
                "end_frame": int,
                "frame_rate": float (如果存在)
            }
        """
        # print(video_id,self.video_frames)
        if video_id not in self.video_frames:
            raise KeyError(f"video_id {video_id} 不存在")

        frame_ids = [self.frame_id_map[img_id] for img_id in self.video_frames[video_id]]
        return {
            "frame_count": len(frame_ids),
            "start_frame": min(frame_ids),
            "end_frame": max(frame_ids),
            "frame_rate": self.video_id_map.get(video_id, {}).get('fps', None)
        }

    def assign_track_ids(self) -> None:
        """为所有标注分配 track_id（按视频分组）"""
        if not hasattr(self, 'frame_id_map'):
            raise ValueError("请先调用 build_frame_sequence() 构建帧序列")
        if not hasattr(self, 'video_frames'):
            raise ValueError("请先调用 assign_video_ids() 分配 video_id")

        # 按视频分组处理
        objects_count=1
        objects_list=[]
        for video_id, frame_list in self.video_frames.items():
            # print(video_id,frame_list)
            # 按帧顺序处理标注
            # for frame_id in sorted(self.frame_id_map[img_id] for img_id in frame_list):
            for frame_id in sorted(frame_list):
                # 获取当前帧的图片 ID 和标注
                image_id = frame_id
                img={}
                for item in self.data['images']:
                    if item['id']==image_id:
                        img=item
                        break
                # print(video_id,image_id)
                detections = [
                    ann for ann in self.data['annotations']
                    if ann['image_id'] == image_id
                    ]
                # 更新跟踪器状态
                updated_detections,objects_count,objects_list= self.tracker.update(video_id,
                                                                      image_id,
                                                                      detections,
                                                                      img,
                                                                      objects_count,
                                                                      objects_list)

                # 更新标注中的 track_id
                for det in updated_detections:
                    ann = next(
                        ann for ann in self.data['annotations']
                        if ann['id'] == det['id']
                    )
                    ann['track_id'] = det['track_id']


# 使用示例
if __name__ == "__main__":
    # 初始化编辑器


    file_dir=os.path.join("../data", "MVA", "annotations")
    file_name="split_train_coco"
    # file_name="single"
    file_path=os.path.join(file_dir,file_name+".json")



    print("start transform")

    start_time=time.time()
    editor = COCOEditor(file_path)

    # 构建帧序列 (按文件名数字部分排序)
    editor.build_frame_sequence(sort_key='file_name')

    # 验证结果
    # sample_img = editor.data['images'][0]
    # print(f"Image {sample_img['id']} 的帧号: {sample_img['frame_id']}")

    # 获取时序相邻帧
    # context_frames = editor.get_adjacent_frames(image_id=1024, n_prev=2, n_next=2)
    # print(f"上下文帧: {context_frames}")

    editor.assign_video_ids()
    editor.build_frame_sequence_by_video(sort_key='file_name')
    editor.assign_track_ids()

    # 获取视频信息
    # video_info = editor.get_video_info(video_id="6")
    # print(f"视频 video1 信息: {video_info}")

    # 保存含frame_id的标注文件
    editor.save(f"{os.path.join(file_dir,file_name)}_frameid_trackid.json")

    end_time=time.time()
    print(f"Spend {end_time-start_time} s")