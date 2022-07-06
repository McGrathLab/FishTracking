"""
    Modified from "SORT: A Simple, Online and Realtime Tracker" and its
    derivative "ClassSort" by Tucker Lancaster. Original Docstrings below.

    =====================================================================

    MINOR MODIFICATION FOR ClassySORT:
    In the original implementation of SORT,
    it threw away the object classification category information
    For example, (0: person, 1: bike, etc.)

    I needed to keep that information for use in `Watchout`,
    so I added a `detclass` attribute to the `KalmanBoxTracker` object
    which stores YOLO detection object class information.
    With this modification, SORT returns data in the format:
    `[x_left_top, y_left_top, x_right_bottom, y_right_bottom, object_category, object_identification]`

    =====================================================================

    Original Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import tarfile

np.random.seed(0)

IMG_W = 1296
IMG_H = 972



def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # include detclass as an attribute to store the detection classification
        self.detclass = bbox[5]

    def update(self, bbox):
        """
    Updates the state vector with observed bbox.
    """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.detclass = bbox[5]

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.detclass = np.NaN
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        arr_detclass = np.expand_dims(np.array([self.detclass]), 0)

        arr_u_dot = np.expand_dims(self.kf.x[4], 0)
        arr_v_dot = np.expand_dims(self.kf.x[5], 0)
        arr_s_dot = np.expand_dims(self.kf.x[6], 0)

        return np.concatenate((convert_x_to_bbox(self.kf.x), arr_detclass, arr_u_dot, arr_v_dot, arr_s_dot), axis=1)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of
    1. matches,
    2. unmatched_detections
    3. unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=None):
        """
    Params:
    dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for
    frames without detections). Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        if dets is None:
            dets = np.empty((0, 6))
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(np.hstack((dets[i,:], np.array([0]))))
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))


"""new code begins here"""


def yolodet_to_sortdet(det):
    """converts a detection of the form [class, x_center, y_center, width, height, score] to the form
  [x1, x2, y1, y2, score, class] """
    det = [float(d) for d in det]
    scaled_det = [det[0], det[1] * IMG_W, det[2] * IMG_H, det[3] * IMG_W, det[4] * IMG_H, det[5]]
    return np.array([(scaled_det[1] - scaled_det[3]) / 2,
                     scaled_det[2] - scaled_det[4] / 2,
                     scaled_det[1] + scaled_det[3] / 2,
                     scaled_det[2] + scaled_det[4] / 2,
                     scaled_det[5], scaled_det[0]])


def update_outfile(trackers, frame, file_obj):
    """
  adds new rows to the output file
  :param trackers: nx5 array, where n is equal to the number of active tracks. This object is returned by Sort.update
  :type trackers: np.ndarray
  :param yolodets: list of the original detections, in yolo format (class, xc, yc, w, h, score)
  :type yolodets: list(list(float | int))
  :param frame: current frame number
  :type frame: int
  :param file_obj: file object that we are writing to
  :type file_obj: TextIOWrapper or equivalent open stream
  """
    for t in trackers:
        xc = ((t[0] + t[2]) / 2) / IMG_W
        yc = ((t[1] + t[3]) / 2) / IMG_H
        w = (t[2] - t[0]) / IMG_W
        h = (t[3] - t[1]) / IMG_H
        class_id = t[4]
        u_dot, v_dot, s_dot = t[5], t[6], t[7]
        track_id = t[8]
        print(f'{track_id}, {frame}, {xc}, {yc}, {w}, {h}, {class_id}, {u_dot}, {v_dot}, {s_dot}', file=file_obj)


def run_sort(infile, min_track_len=90, max_age=5, min_hits=3):
    """

  :param infile: path to the input tarfile, containing the inferences as .txt files
  :type infile: str
  :return: the dataframe that was written to the outfile
  :rtype: pd.DataFrame
  """
    outfile_path = infile.replace('inference.tar', 'tracks.csv')
    outfile = open(outfile_path, 'w')
    outfile.write('track_id,frame,xc,yc,w,h,class_id,u_dot,v_dot,s_dot\n')
    tracker = Sort(max_age=max_age, min_hits=min_hits)
    curr_frame = 0
    with tarfile.open(infile) as tar:
        for member in tar:
            f = tar.extractfile(member)
            try:
                frame = int(member.name.split('_')[-1].split('.')[0])
            except ValueError:
                pass
            else:
                while frame > curr_frame:
                    trackers = tracker.update()
                    update_outfile(trackers, curr_frame, outfile)
                    if not curr_frame % 1000:
                        print(f'frame {curr_frame} processed')
                    curr_frame += 1
                yolodets = [[float(val) for val in l.decode('utf-8').strip().split(' ')] for l in f.readlines()]
                sortdets = np.array([yolodet_to_sortdet(yd) for yd in yolodets])
                trackers = tracker.update(sortdets)
                update_outfile(trackers, curr_frame, outfile)
                if not curr_frame % 1000:
                    print(f'frame {curr_frame} processed')
                curr_frame += 1

    outfile.close()
    tar.close()

    df = pd.read_csv(outfile_path, index_col=0)
    df = df[df.index.value_counts(sort=False) > min_track_len]
    idx = df.index.to_numpy().astype(int)
    for i, j in enumerate(np.unique(idx)):
        idx[idx == j] = i
    df.index = idx
    df.index.rename('track_id', inplace=True)
    df = df.sort_values(by=['track_id', 'frame'])
    df.to_csv(outfile_path, index_label='track_id')
    return df


infile = '../data/MC_singlenuc62_3_Tk65_060220/MC_singlenuc62_3_Tk65_060220_inference.tar'
df = run_sort(infile)
