from collections import defaultdict

class COCOTracker:
    def __init__(self, max_missed_frames: int = 3, iou_threshold: float = 0.05):
        """
        多目标跟踪器（按视频分组）
        参数:
            max_missed_frames: 目标最大丢失帧数
            iou_threshold: 关联阈值 (IoU)
        """
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.tracks = defaultdict(dict)  # {video_id: {track_id: {'last_seen': frame_id, 'bbox': [x, y, w, h]}}}
        self.next_track_ids = defaultdict(int)  # {video_id: next_track_id}

        #
        self.obj_amount=0

    def iou(self, box1, box2):
        """计算两个边界框的 IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算相交区域
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        return inter_area / (box1_area + box2_area - inter_area)

    def update(self,
               video_id: str,
               frame_id: int,
               detections: [dict],
               frame_image:dict,
               obj_count:int,
               obj_list:[]) -> [dict]:
        """
        更新跟踪器状态并分配 track_id
        参数:
            video_id: 视频 ID
            frame_id: 当前帧 ID
            detections: 当前帧的目标检测结果 [{'bbox': [x, y, w, h], ...}]
        返回:
            更新后的检测结果 [{'bbox': [x, y, w, h], 'track_id': int}]
        """
        # 初始化匹配矩阵
        matched_tracks = set()
        updated_detections = []

        # 遍历当前帧的检测结果
        # print(video_id,frame_id,obj_list)
        frame_seq = frame_image["frame_id"]
        obj_amount = len(detections)

        if frame_seq == 1:
            self.obj_amount += len(detections)
            print(frame_image,len(obj_list),len(detections),self.obj_amount)
        for det in detections:
            #TODO here frame_id is the global "id" of images rather than "frame sequence"
            if frame_seq==1:
                obj_list = []
                obj_id=obj_count
                obj_list.append(obj_id)
                det['track_id'] = obj_id
                updated_detections.append(det)
                self.tracks[obj_id] = {
                    'bbox': det['bbox'],
                    'last_seen': 1
                }
                obj_count += 1
                continue
            else:
                for obj_id in obj_list:
                    #TODO use global id rather than video_id->obj_id, while obj_id is now global
                    if frame_seq - self.tracks[obj_id]['last_seen'] > self.max_missed_frames:
                        obj_list.remove(obj_id)
                        continue
                    elif self.iou(det['bbox'], self.tracks[obj_id]['bbox']) >= self.iou_threshold:
                        det['track_id'] = obj_id
                        self.tracks[obj_id]['bbox'] = det['bbox']
                        self.tracks[obj_id]['last_seen'] = frame_seq
                        updated_detections.append(det)
                        obj_amount -= 1
                        break

                if obj_amount>0:
                    obj_id=obj_count
                    obj_list.append(obj_id)
                    det['track_id'] = obj_id
                    updated_detections.append(det)
                    self.tracks[obj_id] = {
                        'bbox': det['bbox'],
                        'last_seen': frame_seq
                    }
                    obj_count += 1
            #
            # if video_id==38 and frame_id<=100:
            #     print(len(updated_detections))
            # print("-----------------------")
            # best_match = None
            # best_iou = self.iou_threshold
            # # print(self.tracks[video_id])
            # # 遍历现有轨迹
            # for track_id, track in self.tracks[video_id].items():
            #     print("track id is",track_id,"track is",track)
            #     if track_id in matched_tracks:
            #         print("jump")
            #         continue  # 跳过已匹配的轨迹
            #
            #     # 计算 IoU
            #     iou_score = self.iou(det['bbox'], track['bbox'])
            #     if iou_score >= best_iou:
            #         best_iou = iou_score
            #         best_match = track_id
            #
            #
            # # 关联成功
            # if best_match is not None:
            #     det['track_id'] = best_match
            #     matched_tracks.add(best_match)
            #     self.tracks[video_id][best_match]['bbox'] = det['bbox']
            #     self.tracks[video_id][best_match]['last_seen'] = frame_id
            # else:
            #     # 新目标
            #     print("new object",self.next_track_ids[video_id])
            #     det['track_id'] = self.next_track_ids[video_id]
            #     self.tracks[video_id][self.next_track_ids[video_id]] = {
            #         'bbox': det['bbox'],
            #         'last_seen': frame_id
            #     }
            #     self.next_track_ids[video_id] += 1
            #
            # updated_detections.append(det)

        # 清理丢失的轨迹
        # lost_tracks = [
        #     track_id for track_id, track in self.tracks[video_id].items()
        #     if frame_id - track['last_seen'] > self.max_missed_frames
        # ]
        # for track_id in lost_tracks:
        #     del self.tracks[video_id][track_id]

        return updated_detections,obj_count,obj_list