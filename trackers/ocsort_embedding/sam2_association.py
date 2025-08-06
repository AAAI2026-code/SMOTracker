
from __future__ import print_function

import pdb
import pickle

import cv2
import torch
import torchvision

import numpy as np
from .association_yolo import *
from .embedding import EmbeddingComputer
from .assignment import *
from .nn_matching import *
from .sam2_tracker import *
from sam2_functions import *

class OCSort(object):
    def __init__(
        self,
        det_thresh,
        alpha_gate,
        gate,
        gate2,
        max_age=35,
        min_hits=2,
        iou_threshold=0.9,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.75,
        new_kf_off=False,
        predictor_dict=None,
        **kwargs,
    ):

        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb

        #
        if predictor_dict is not None:
            self.predictor=predictor_dict["predictor"]
            self.pre_seg=predictor_dict["segments"]
            self.inference_state=predictor_dict["infer_state"]
        else:
            raise TypeError("predictor should not be None.")

        self.max_id=0
        self.gap_size=predictor_dict["buffer_size"]
        #
        self.embedder = EmbeddingComputer(
            kwargs["args"].dataset, kwargs["args"].test_dataset)
        self.new_kf_off = new_kf_off

        self.min_hits = min_hits

        self.sigma = 5

        self.alpha_gate = alpha_gate
        self.gate = gate
        self.gate2 = gate2

    def extract_detections(self, output_results, img_tensor, img_numpy, tag=None):

        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        dets = np.concatenate(
            (bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        scale = min(img_tensor.shape[1] / img_numpy.shape[0],
                    img_tensor.shape[2] / img_numpy.shape[1])

        dets[:, :4] /= scale

        inds_low = scores > 0.2 
        inds_high = scores <= 0.3
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = scores > 0.3
        dets_one = dets[remain_inds]
        return dets_one, dets_second

    def generate_embs(self, dets, img_numpy, tag):
        dets_embs = np.ones((dets.shape[0], 1))
        if dets.shape[0] != 0:
            dets_embs = self.embedder.compute_embedding(
                img_numpy, dets[:, :4], tag)
        return dets_embs

    def get_pred_loc_from_exist_tracks(self):
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)

        # velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
        #     (0, 0)) for trk in self.trackers])
        # speeds = np.array(
        #     [trk.speed if trk.speed is not None else 0 for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(
            trk.observations, trk.age, self.delta_t) for trk in self.trackers])
        return trks,last_boxes, k_observations, trk_embs

    def sam2_iou_third_associate(self, dets, last_boxes, unmatched_dets, unmatched_trks, iou_threshold):
        left_dets = dets[unmatched_dets]
        left_trks = last_boxes[unmatched_trks]

        iou_left = self.asso_func(left_dets, left_trks)
        iou_left = np.array(iou_left)

        matches = []
        if iou_left.max() > iou_threshold:

            rematched_indices = linear_assignment(-iou_left)

            to_remove_det_indices = []
            to_remove_trk_indices = []
            for m in rematched_indices:
                det_ind, trk_ind = unmatched_dets[m[0]
                ], unmatched_trks[m[1]]
                if iou_left[m[0], m[1]] < iou_threshold:
                    continue
                m_ = np.array([trk_ind, det_ind,0])
                matches.append(m_.reshape(1, 3))
                to_remove_det_indices.append(det_ind)
                to_remove_trk_indices.append(trk_ind)
            unmatched_dets = np.setdiff1d(
                unmatched_dets, np.array(to_remove_det_indices))
            unmatched_trks = np.setdiff1d(
                unmatched_trks, np.array(to_remove_trk_indices))

        matches = np.array(matches)
        if (len(matches) == 0):
            matches = np.empty((0, 3), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        matches = np.hstack(
            (matches, np.zeros(matches.shape[0]).reshape(-1, 1)))

        return (matches, unmatched_trks, unmatched_dets)

    def update(self, output_results, img_tensor, img_numpy, tag, metric, two_round_off,frame_idx):
        if output_results is None:
            return np.empty((0, 5))
        self.frame_count=frame_idx
        if len(self.inference_state["obj_ids"]) > 0:
            self.max_id = max(self.max_id, max(self.inference_state["obj_ids"]))
        else:
            pass

        dets_one, dets_second = self.extract_detections(
            output_results, img_tensor, img_numpy, tag)

        dets_one_embs = self.generate_embs(dets_one, img_numpy, tag+"@one")


        dets_second_embs = self.generate_embs(
            dets_second, img_numpy, tag+"@second")
        
        ret= self.sam2_assign_cascade(dets_one, dets_one_embs,
                                  dets_second, dets_second_embs, metric, two_round_off)
        return ret

    def sam2_level_matching(self, depth,
                       track_indices,
                       detection_one_indices,
                       detection_second_indices,
                       trk_embs,
                       dets_one_embs,
                       dets_second_embs,
                       trks,
                       dets_one,
                       dets_second,
                       k_observations,
                       last_boxes,
                       metric,
                       two_round_off,
                       ):

        if track_indices is None:
            track_indices = np.arange(len(trks))
        if detection_one_indices is None:
            detection_one_indices = np.arange(len(dets_one))
        if detection_second_indices is None:
            detection_second_indices = np.arange(len(dets_second))

        if len(detection_one_indices) != 0 and len(track_indices) != 0:

            gate = self.gate if depth == 0 else self.gate2
            appearance_pre_assign, emb_cost = sam2_appearance_associate(
                dets_one_embs,
                trk_embs,
                dets_one, trks,
                track_indices, detection_one_indices,
                gate, self.iou_threshold, metric)

        #
            if depth == 0:
                iou_pre_assign = sam2_associate(
                    dets_one,
                    trks,
                    dets_one_embs,
                    trk_embs,
                    self.iou_threshold,
                    k_observations,
                    self.w_association_emb,
                    track_indices, gate, metric, two_round_off
                )
            else:
                iou_pre_assign = (np.empty(
                    (0, 4), dtype=int), [], [])

            if not two_round_off:
                appearance_pre_assign = iou_pre_assign

            matched_one_1, unmatched_trks_1, unmatched_dets_one_1 = min_cost_matching(
                iou_pre_assign, appearance_pre_assign, self.alpha_gate, two_round_off)

        else:
            matched_one_1, unmatched_trks_1, unmatched_dets_one_1 = np.empty(
                (0, 2), dtype=int), track_indices, detection_one_indices

        unmatched_trks = unmatched_trks_1
        if len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            iou_threshold = 0.9
            gate = self.gate2
            if depth == 0:
                iou_pre_assign = self.sam2_iou_second_associate(trks, dets_second, unmatched_trks, iou_threshold)
                appearance_pre_assign = iou_pre_assign

            else:
                left_trk_embs = [trk_embs[i] for i in unmatched_trks]
                left_trks = trks[unmatched_trks]

                appearance_pre_assign, _ = sam2_appearance_associate(
                    dets_second_embs,
                    left_trk_embs,
                    dets_second, left_trks,
                    unmatched_trks, detection_second_indices,
                    gate, self.iou_threshold)
                iou_pre_assign = appearance_pre_assign

            matches_second, unmatched_trks_second, unmatched_dets_second = min_cost_matching(
                iou_pre_assign, appearance_pre_assign, self.alpha_gate)
        else:

            matches_second = np.empty((0, 2), dtype=int)
            unmatched_trks_second = unmatched_trks
            unmatched_dets_second = detection_second_indices

        unmatched_trks_2 = unmatched_trks_second

        if unmatched_dets_one_1.shape[0] > 0 and unmatched_trks_2.shape[0] > 0:
            iou_threshold = self.iou_threshold

            gate = self.gate2
            if depth == 0:

                iou_pre_assign = self.sam2_iou_third_associate(
                    dets=dets_one,
                    last_boxes=last_boxes,
                    unmatched_dets=unmatched_dets_one_1,
                    unmatched_trks=unmatched_trks_2,
                    iou_threshold=iou_threshold)

                appearance_pre_assign = iou_pre_assign

            else:
                left_dets_embs = dets_one_embs[unmatched_dets_one_1]
                left_trks_embs = [trk_embs[i] for i in unmatched_trks_2]
                left_dets = dets_one[unmatched_dets_one_1]
                left_trks = trks[unmatched_trks_2]
                appearance_pre_assign, _ = sam2_appearance_associate(
                    left_dets_embs,
                    left_trks_embs,
                    left_dets, left_trks,
                    unmatched_trks_2, unmatched_dets_one_1,
                    gate, iou_threshold)

                iou_pre_assign = appearance_pre_assign

            matched_one_2, unmatched_trks_3, unmatched_dets_one_2 = min_cost_matching(
                iou_pre_assign, appearance_pre_assign, self.alpha_gate)
        else:

            matched_one_2 = np.empty((0, 2), dtype=int)
            unmatched_trks_3 = unmatched_trks_2
            unmatched_dets_one_2 = unmatched_dets_one_1

        if len(matched_one_1) and len(matched_one_2):
            matches_one = np.concatenate(
                (matched_one_1, matched_one_2), axis=0)
        elif len(matched_one_2):
            matches_one = matched_one_2
        elif len(matched_one_1):
            matches_one = matched_one_1
        else:
            matches_one = np.empty((0, 2), dtype=int)

        unmatched_trks = unmatched_trks_3
        unmatched_dets_one = unmatched_dets_one_2

        return matches_one, matches_second, unmatched_trks, unmatched_dets_one, unmatched_dets_second

    def sam2_iou_second_associate(self, trks, dets_second, unmatched_trks, iou_threshold):
        u_trks = trks[unmatched_trks]

        iou_left = self.asso_func(dets_second, u_trks)
        iou_left = np.array(iou_left)

        matched_2, unmatched_trks_2 = [], unmatched_trks
        if iou_left.max() > iou_threshold:

            matched_indices = linear_assignment(-iou_left)

            to_remove_trk_indices = []
            for m in matched_indices:
                det_ind, trk_ind = m[0], unmatched_trks[m[1]]

                if iou_left[m[0], m[1]] < iou_threshold:
                    continue
                m_ = np.array([trk_ind, det_ind, 0])
                matched_2.append(m_.reshape(1, 3))

                to_remove_trk_indices.append(trk_ind)

            unmatched_trks_2 = np.setdiff1d(
                unmatched_trks, np.array(to_remove_trk_indices))

        matched_2 = np.array(matched_2)
        if (len(matched_2) == 0):
            matched_2 = np.empty((0, 3), dtype=int)
        else:
            matched_2 = np.concatenate(matched_2, axis=0)

        matched_2 = np.hstack(
            (matched_2, np.zeros(matched_2.shape[0]).reshape(-1, 1)))

        return (matched_2, unmatched_trks_2, np.array([]))

    def map_origin_ind(self, track_indices_l,
                       unmatched_dets,
                       matches_l, unmatched_trks_l, unmatched_dets_l):

        track_indices_l = np.array(track_indices_l)
        unmatched_dets = np.array(unmatched_dets)

        if matches_l.shape[0]:
            matches_l_ = matches_l[:, :2].astype(np.int_)
            matches_l[:, 0] = track_indices_l[matches_l_[:, 0]]
            matches_l[:, 1] = unmatched_dets[matches_l_[:, 1]]

        if len(unmatched_trks_l):
            unmatched_trks_l[:] = track_indices_l[unmatched_trks_l[:]]

        if len(unmatched_dets_l):
            unmatched_dets_l[:] = unmatched_dets[unmatched_dets_l[:]]

        return matches_l, unmatched_trks_l, unmatched_dets_l

    def sam2_assign_cascade(self, dets_one, dets_one_embs,
                       dets_second, dets_second_embs, metric, two_round_off):
        ret = []
        trks,last_boxes, k_observations, trk_embs = \
            self.get_pred_loc_from_exist_tracks()
        track_indices = list(range(len(trks)))
        detection_one_indices = list(range(len(dets_one)))
        detection_second_indices = list(range(len(dets_second)))
        unmatched_dets_one = detection_one_indices

        unmatched_dets_second = detection_second_indices

        matched_one = np.empty((0, 5), dtype=int)
        matched_second = np.empty((0, 5), dtype=int)

        unmatched_trks_l = []

        for depth in range(1):
            if len(unmatched_dets_one) == 0 and len(unmatched_dets_second) == 0:
                break

            track_indices_l = [
                k for k in track_indices if self.trackers[k].time_since_update >= 1 + depth]
            if len(track_indices_l) == 0:
                continue

            matches_one_l, matches_second_l, unmatched_trks_l, unmatched_dets_one_l, unmatched_dets_second_l = \
                self.sam2_level_matching(depth, None, None, None,
                                    trk_embs[track_indices_l],
                                    dets_one_embs[unmatched_dets_one],
                                    dets_second_embs[unmatched_dets_second],
                                    trks[track_indices_l],
                                    dets_one[unmatched_dets_one],
                                    dets_second[unmatched_dets_second],
                                    k_observations[track_indices_l],
                                    last_boxes[track_indices_l], metric, two_round_off)
            matches_one_l, unmatched_trks_l, unmatched_dets_one_l = \
                self.map_origin_ind(track_indices_l,
                                    unmatched_dets_one,
                                    matches_one_l, unmatched_trks_l, unmatched_dets_one_l)
            matches_second_l, _, unmatched_dets_second_l = \
                self.map_origin_ind(track_indices_l,
                                    unmatched_dets_second,
                                    matches_second_l, [], unmatched_dets_second_l)
            unmatched_dets_one = list(unmatched_dets_one_l)
            unmatched_dets_second = list(unmatched_dets_second_l)
            if len(matches_one_l):
                matched_one = np.concatenate(
                    (matched_one, matches_one_l), axis=0)
            if len(matches_second_l):
                matched_second = np.concatenate(
                    (matched_second, matches_second_l), axis=0)
                
        unmatched_trks = list(
            set(track_indices) - set(matched_one[:, 0]) - set(matched_second[:, 0]))

        for m in matched_one:
            self.trackers[int(m[0])].update(dets_one[int(m[1]), :])
            self.trackers[int(m[0])].update_emb(
                dets_one_embs[int(m[1])])

        for m in matched_second:
            self.trackers[int(m[0])].update(dets_second[int(m[1]), :])
            self.trackers[int(m[0])].update_emb(
                dets_second_embs[int(m[1])])

        for m in unmatched_trks:
            self.trackers[m].update(None)
        for i in unmatched_dets_one:
            self.max_id=self.max_id+1
            trk = SAM2Tracker(
                bbox=dets_one[i,:4],count=self.max_id, seg_memory=self.pre_seg, delta_t=self.delta_t, emb=dets_one_embs[i],
                new_kf=not self.new_kf_off
            )
            self.trackers.append(trk)
            _, _= add_new_prompt(
                inference_state=self.inference_state,
                predictor=self.predictor,
                frame_id=0,
                obj_id=self.max_id,
                object_prompt=dets_one[i,:4]
            )

        for i in unmatched_dets_second:
            self.max_id = self.max_id+1
            trk = SAM2Tracker(
                bbox=dets_second[i, :4], count=self.max_id, seg_memory=self.pre_seg, delta_t=self.delta_t,emb=dets_one_embs[i],
                new_kf=not self.new_kf_off
            )
            self.trackers.append(trk)
            _, _ = add_new_prompt(
                inference_state=self.inference_state,
                predictor=self.predictor,
                frame_id=0,
                obj_id=self.max_id,
                object_prompt=dets_one[i, :4]
            )
        #
        ret = self.final_process(matched_one)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def get_row_by_id(self, array, id):
        for row in array:
            if row[0] == id:
                return row[4]
        return None

    def final_process(self, matched_one):
        i = len(self.trackers)

        ret = []
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

            i -= 1

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return ret

    def assign_(self, dets, dets_embs, dets_alpha):
        ret = []

        trks, velocities, speeds, last_boxes, k_observations, trk_embs = \
            self.get_pred_loc_from_exist_tracks()

        track_indices = list(range(len(trks)))
        detection_indices = list(range(len(dets)))

        if len(detection_indices) != 0 and len(track_indices) != 0:

            tracks_info = {}
            alpha = metric_gaussian_motion(speeds, sigma=self.sigma)
            for i, idx in enumerate(track_indices):
                tracks_info[idx] = alpha[i]

            appearance_pre_assign, emb_cost = appearance_associate(
                dets_embs,
                trk_embs,
                dets, trks,
                track_indices, detection_indices, tracks_info,
                self.gate, self.iou_threshold)

            iou_pre_assign = associate(
                dets,
                trks,
                dets_embs,
                trk_embs,
                emb_cost,
                self.iou_threshold,
                velocities,
                k_observations,
                self.inertia,
                self.w_association_emb,
                track_indices, tracks_info,
            )

            matched, unmatched_trks, unmatched_dets = min_cost_matching(
                iou_pre_assign, appearance_pre_assign, self.alpha_gate)

            for m in matched:

                self.trackers[m[0]].update(dets[m[1], :])
                self.trackers[m[0]].update_emb(
                    dets_embs[m[1]])

        else:
            matched, unmatched_trks, unmatched_dets = np.empty(
                (0, 2), dtype=int), np.empty((0, 5), dtype=int), np.arange(len(detection_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            iou_pre_assign = self.sam2_iou_third_associate(
                dets, last_boxes, unmatched_dets, unmatched_trks, self.iou_threshold)

            left_dets_embs = dets_embs[unmatched_dets]
            left_trks_embs = [trk_embs[i] for i in unmatched_trks]
            left_dets = dets[unmatched_dets]
            left_trks = trks[unmatched_trks]

            appearance_pre_assign, emb_cost = appearance_associate(
                left_dets_embs,
                left_trks_embs,
                left_dets, left_trks,
                unmatched_trks, unmatched_dets,
                tracks_info,
                self.gate, self.iou_threshold)

            matched, unmatched_trks, unmatched_dets = min_cost_matching(
                iou_pre_assign, appearance_pre_assign, self.alpha_gate)

            for m in matched:
                self.trackers[int(m[0])].update(dets[int(m[1]), :])
                self.trackers[int(m[0])].update_emb(
                    dets_embs[int(m[1])])

        for m in unmatched_trks:
            self.trackers[m].update(None)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :], delta_t=self.delta_t, emb=dets_embs[i], alpha=dets_alpha[i], new_kf=not self.new_kf_off
            )
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:

                d = trk.last_observation[:4]

            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):

                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):

        self.embedder.dump_cache()

