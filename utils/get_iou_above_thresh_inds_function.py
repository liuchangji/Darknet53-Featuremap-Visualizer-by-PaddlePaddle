import numpy as np
# 挑选出跟真实框IoU大于阈值的预测框
def get_iou_above_thresh_inds(pred_box, gt_boxes, iou_threshold):
    batchsize = pred_box.shape[0]
    num_rows = pred_box.shape[1]
    num_cols = pred_box.shape[2]
    num_anchors = pred_box.shape[3]
    ret_inds = np.zeros([batchsize, num_rows, num_cols, num_anchors])
    for i in range(batchsize):
        pred_box_i = pred_box[i]
        gt_boxes_i = gt_boxes[i]
        for k in range(len(gt_boxes_i)): #gt in gt_boxes_i:
            gt = gt_boxes_i[k]
            gtx_min = gt[0] - gt[2] / 2.
            gty_min = gt[1] - gt[3] / 2.
            gtx_max = gt[0] + gt[2] / 2.
            gty_max = gt[1] + gt[3] / 2.
            if (gtx_max - gtx_min < 1e-3) or (gty_max - gty_min < 1e-3):
                continue
            x1 = np.maximum(pred_box_i[:, :, :, 0], gtx_min)
            y1 = np.maximum(pred_box_i[:, :, :, 1], gty_min)
            x2 = np.minimum(pred_box_i[:, :, :, 2], gtx_max)
            y2 = np.minimum(pred_box_i[:, :, :, 3], gty_max)
            intersection = np.maximum(x2 - x1, 0.) * np.maximum(y2 - y1, 0.)
            s1 = (gty_max - gty_min) * (gtx_max - gtx_min)
            s2 = (pred_box_i[:, :, :, 2] - pred_box_i[:, :, :, 0]) * (pred_box_i[:, :, :, 3] - pred_box_i[:, :, :, 1])
            union = s2 + s1 - intersection
            iou = intersection / union
            above_inds = np.where(iou > iou_threshold)
            ret_inds[i][above_inds] = 1
    ret_inds = np.transpose(ret_inds, (0,3,1,2))
    return ret_inds.astype('bool')


def label_objectness_ignore(label_objectness, iou_above_thresh_indices):
    # 注意：这里不能简单的使用 label_objectness[iou_above_thresh_indices] = -1，
    #         这样可能会造成label_objectness为1的那些点被设置为-1了
    #         只有将那些被标注为0，且与真实框IoU超过阈值的预测框才被标注为-1
    negative_indices = (label_objectness < 0.5)
    ignore_indices = negative_indices * iou_above_thresh_indices
    label_objectness[ignore_indices] = -1
    return label_objectness