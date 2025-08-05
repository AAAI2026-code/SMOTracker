import numpy as np
from .association_yolo import *

ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_bbox_to_z_new(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox_new(x):
    x, y, w, h = x.reshape(-1)[:4]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)


def convert_x_to_bbox(x, score=None):

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6

    level = 1-iou_batch(np.array([bbox1]), np.array([bbox2]))[0][0]
    return speed / norm, level


def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
    Q = np.diag(
        (
            (p * w) ** 2,
            (p * h) ** 2,
            (p * w) ** 2,
            (p * h) ** 2,
            (v * w) ** 2,
            (v * h) ** 2,
            (v * w) ** 2,
            (v * h) ** 2,
        )
    )
    return Q


def new_kf_measurement_noise(w, h, m=1 / 20):
    w_var = (m * w) ** 2
    h_var = (m * h) ** 2
    R = np.diag((w_var, h_var, w_var, h_var))
    return R

class SAM2Tracker(object):
    def __init__(self,bbox, count:int,seg_memory,delta_t=3, orig=False, emb=None, alpha=0, new_kf=False):
        # the structure of seg_memory is like predictor[frame][object_id]
        self.predictor=seg_memory
        self.new_kf = new_kf


        self.time_since_update = 0
        self.id = count
        self.state=bbox

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.history_observations = []
        self.observations = dict()
        self.sam2_seg=seg_memory

        self.delta_t = delta_t
        self.emb = emb
        self.frozen = False
        self.budget = 30
        self.emb_ind = 0
        self.emb_ind += 1

    def update(self,bbox):
        if bbox is not None:
            self.frozen = False

            if self.last_observation.sum() >= 0:
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation

                # self.velocity, self.speed = speed_direction(previous_box, bbox)

            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.state=bbox
        else:
            self.frozen = True

    def update_emb(self, emb, alpha=0):

        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):

        return self.emb

    def predict(self):
        seg_max_frame=max(self.sam2_seg.keys())
        in_frame_objects=self.sam2_seg[seg_max_frame]
        bbox = []
        if self.id in in_frame_objects.keys():
            id_mask=in_frame_objects[self.id]
            # transfer from mask to bbox
            id_mask=id_mask.squeeze()
            non_zero_indices = np.argwhere(id_mask)

            print("size:",non_zero_indices.size )
            if non_zero_indices.size > 0:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                bbox = [x_min, y_min, x_max, y_max]
            else:
                bbox = [0, 0, 0, 0]

        else:
            bbox=self.last_observation
            # bbox=bbox.tolist()
            # bbox=[int(x) for x in bbox]
            #
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        print(bbox)
        return bbox

    def get_state(self):

        return self.state

    # def mahalanobis(self, bbox):
    #
    #     return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))