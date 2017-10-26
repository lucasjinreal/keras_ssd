import numpy as np


def iou(boxes1, boxes2, coords='centroids'):
    if len(boxes1.shape) > 2: raise ValueError(
        "boxes1 must have rank either 1 or 2, but has rank {}.".format(len(boxes1.shape)))
    if len(boxes2.shape) > 2: raise ValueError(
        "boxes2 must have rank either 1 or 2, but has rank {}.".format(len(boxes2.shape)))

    if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("It must be boxes1.shape[1] == boxes2.shape[1] == 4, "
                         "but it is boxes1.shape[1] == {}, boxes2.shape[1] == {}.".format(boxes1.shape[1],
                                                                                          boxes2.shape[1]))

    if coords == 'centroids':
        # TODO: Implement a version that uses fewer computation steps (that doesn't need conversion)
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2minmax')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2minmax')
    elif coords != 'minmax':
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    intersection = np.maximum(0, np.minimum(boxes1[:, 1], boxes2[:, 1]) -
                              np.maximum(boxes1[:, 0], boxes2[:, 0])) * \
                   np.maximum(0, np.minimum(boxes1[:, 3], boxes2[:, 3]) - np.maximum(boxes1[:, 2], boxes2[:, 2]))
    union = (boxes1[:, 1] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 2]) + (boxes2[:, 1] - boxes2[:, 0]) * (
        boxes2[:, 3] - boxes2[:, 2]) - intersection

    return intersection / union


def convert_coordinates(tensor, start_index, conversion='minmax2centroids'):
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind + 1]) / 2.0  # Set cx
        tensor1[..., ind + 1] = (tensor[..., ind + 2] + tensor[..., ind + 3]) / 2.0  # Set cy
        tensor1[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind]  # Set w
        tensor1[..., ind + 3] = tensor[..., ind + 3] - tensor[..., ind + 2]  # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0  # Set xmin
        tensor1[..., ind + 1] = tensor[..., ind] + tensor[..., ind + 2] / 2.0  # Set xmax
        tensor1[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0  # Set ymin
        tensor1[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0  # Set ymax
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1


def convert_coordinates2(tensor, start_index, conversion='minmax2centroids'):
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        M = np.array([[0.5, 0., -1., 0.],
                      [0.5, 0., 1., 0.],
                      [0., 0.5, 0., -1.],
                      [0., 0.5, 0., 1.]])
        tensor1[..., ind:ind + 4] = np.dot(tensor1[..., ind:ind + 4], M)
    elif conversion == 'centroids2minmax':
        M = np.array([[1., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [-0.5, 0.5, 0., 0.],
                      [0., 0., -0.5, 0.5]])  # The multiplicative inverse of the matrix above
        tensor1[..., ind:ind + 4] = np.dot(tensor1[..., ind:ind + 4], M)
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1


def greedy_nms(y_pred_decoded, iou_threshold=0.45, coords='minmax'):
    y_pred_decoded_nms = []
    for batch_item in y_pred_decoded:  # For the labels of each batch item...
        boxes_left = np.copy(batch_item)
        maxima = []  # This is where we store the boxes that make it through the non-maximum suppression
        while boxes_left.shape[0] > 0:  # While there are still boxes left to compare...
            maximum_index = np.argmax(
                boxes_left[:, 1])  # ...get the index of the next box with the highest confidence...
            maximum_box = np.copy(boxes_left[maximum_index])  # ...copy that box and...
            maxima.append(maximum_box)  # ...append it to `maxima` because we'll definitely keep it
            boxes_left = np.delete(boxes_left, maximum_index, axis=0)  # Now remove the maximum box from `boxes_left`
            if boxes_left.shape[0] == 0: break  # If there are no boxes left after this step, break. Otherwise...
            similarities = iou(boxes_left[:, 2:], maximum_box[2:],
                               coords=coords)  # ...compare (IoU) the other left over boxes to the maximum box...
            boxes_left = boxes_left[
                similarities <= iou_threshold]
        y_pred_decoded_nms.append(np.array(maxima))

    return y_pred_decoded_nms


def _greedy_nms(predictions, iou_threshold=0.45, coords='minmax'):
    boxes_left = np.copy(predictions)
    maxima = []  # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0:  # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:, 0])  # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index])  # ...copy that box and...
        maxima.append(maximum_box)  # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0)  # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break  # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:, 1:], maximum_box[1:],
                           coords=coords)  # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[
            similarities <= iou_threshold]
    return np.array(maxima)


def _greedy_nms2(predictions, iou_threshold=0.45, coords='minmax'):
    boxes_left = np.copy(predictions)
    maxima = []  # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0:  # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:, 1])  # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index])  # ...copy that box and...
        maxima.append(maximum_box)  # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0)  # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break  # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:, 2:], maximum_box[2:],
                           coords=coords)  # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[
            similarities <= iou_threshold]
    return np.array(maxima)


def decode_y(y_pred,
             confidence_thresh=0.01,
             iou_threshold=0.45,
             top_k=200,
             input_coords='centroids',
             normalize_coords=False,
             img_height=None,
             img_width=None):
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError(
            "If relative box coordinates are supposed to be converted to absolute coordinates, "
            "the decoder needs the image size in order to decode the predictions, "
            "but `img_height == {}` and `img_width == {}`".format(
                img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:, :, :-8])
    if input_coords == 'centroids':
        y_pred_decoded_raw[:, :, [-2, -1]] = np.exp(y_pred_decoded_raw[:, :, [-2, -1]] * y_pred[:, :, [-2,
                                                                                                       -1]])
        y_pred_decoded_raw[:, :, [-2, -1]] *= y_pred[:, :, [-6,
                                                            -5]]
        y_pred_decoded_raw[:, :, [-4, -3]] *= y_pred[:, :, [-4, -3]] * y_pred[:, :, [-6,
                                                                                     -5]]
        y_pred_decoded_raw[:, :, [-4, -3]] += y_pred[:, :, [-8,
                                                            -7]]
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2minmax')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:, :, -4:] *= y_pred[:, :,
                                         -4:]
        y_pred_decoded_raw[:, :, [-4, -3]] *= np.expand_dims(y_pred[:, :, -7] - y_pred[:, :, -8],
                                                             axis=-1)
        y_pred_decoded_raw[:, :, [-2, -1]] *= np.expand_dims(y_pred[:, :, -5] - y_pred[:, :, -6],
                                                             axis=-1)
        y_pred_decoded_raw[:, :, -4:] += y_pred[:, :, -8:-4]  # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError(
            "Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax' and 'centroids'.")

    if normalize_coords:
        y_pred_decoded_raw[:, :, -4:-2] *= img_width  # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:, :, -2:] *= img_height  # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[
                    -1] - 4  # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = []  # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw:  # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = []  # Store the final predictions for this batch item here
        for class_id in range(1, n_classes):  # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:, [class_id, -4, -3, -2,
                                          -1]]
            threshold_met = single_class[single_class[:,
                                         0] > confidence_thresh]
            if threshold_met.shape[0] > 0:  # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold,
                                     coords='minmax')  # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[
                    1] + 1))
                maxima_output[:, 0] = class_id  # Write the class ID to the first column...
                maxima_output[:, 1:] = maxima  # ...and write the maxima to the other columns...
                pred.append(
                    maxima_output)  # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        pred = np.concatenate(pred, axis=0)
        if pred.shape[
            0] > top_k:
            top_k_indices = np.argpartition(pred[:, 1], kth=pred.shape[0] - top_k, axis=0)[
                            pred.shape[0] - top_k:]  # ...get the indices of the `top_k` highest-score maxima...
            pred = pred[top_k_indices]  # ...and keep only those entries of `pred`...
        y_pred_decoded.append(
            pred)

    return y_pred_decoded


def decode_y2(y_pred,
              confidence_thresh=0.5,
              iou_threshold=0.45,
              top_k='all',
              input_coords='centroids',
              normalize_coords=False,
              img_height=None,
              img_width=None):
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError(
            "If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the "
            "image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(
                img_height, img_width))

    # 1: Convert the classes from one-hot encoding to their class ID
    y_pred_converted = np.copy(y_pred[:, :,
                               -14:-8])
    y_pred_converted[:, :, 0] = np.argmax(y_pred[:, :, :-12],
                                          axis=-1)
    y_pred_converted[:, :, 1] = np.amax(y_pred[:, :, :-12], axis=-1)  # Store the confidence values themselves, too

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    if input_coords == 'centroids':
        y_pred_converted[:, :, [4, 5]] = np.exp(y_pred_converted[:, :, [4, 5]] * y_pred[:, :, [-2,
                                                                                               -1]])
        y_pred_converted[:, :, [4, 5]] *= y_pred[:, :, [-6,
                                                        -5]]
        y_pred_converted[:, :, [2, 3]] *= y_pred[:, :, [-4, -3]] * y_pred[:, :, [-6,
                                                                                 -5]]
        y_pred_converted[:, :, [2, 3]] += y_pred[:, :, [-8,
                                                        -7]]
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='centroids2minmax')
    elif input_coords == 'minmax':
        y_pred_converted[:, :, 2:] *= y_pred[:, :,
                                      -4:]
        y_pred_converted[:, :, [2, 3]] *= np.expand_dims(y_pred[:, :, -7] - y_pred[:, :, -8],
                                                         axis=-1)
        y_pred_converted[:, :, [4, 5]] *= np.expand_dims(y_pred[:, :, -5] - y_pred[:, :, -6],
                                                         axis=-1)
        y_pred_converted[:, :, 2:] += y_pred[:, :, -8:-4]
    else:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    if normalize_coords:
        y_pred_converted[:, :, 2:4] *= img_width  # Convert xmin, xmax back to absolute coordinates
        y_pred_converted[:, :, 4:] *= img_height  # Convert ymin, ymax back to absolute coordinates

    y_pred_decoded = []
    for batch_item in y_pred_converted:  # For each image in the batch...
        boxes = batch_item[
            np.nonzero(batch_item[:, 0])]  # ...get all boxes that don't belong to the background class,...
        boxes = boxes[boxes[:,
                      1] >= confidence_thresh]
        if iou_threshold:  # ...if an IoU threshold is set...
            boxes = _greedy_nms2(boxes, iou_threshold=iou_threshold,
                                 coords='minmax')  # ...perform NMS on the remaining boxes.
        if top_k != 'all' and boxes.shape[0] > top_k:  # If we have more than `top_k` results left at this point...
            top_k_indices = np.argpartition(boxes[:, 1], kth=boxes.shape[0] - top_k, axis=0)[
                            boxes.shape[0] - top_k:]  # ...get the indices of the `top_k` highest-scoring boxes...
            boxes = boxes[top_k_indices]  # ...and keep only those boxes...
        y_pred_decoded.append(
            boxes)

    return y_pred_decoded


class SSDBoxEncoder:
    '''
    A class to transform ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an SSD model, and to transform predictions of the SSD model back
    to the original format of the input labels.

    In the process of encoding ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 limit_boxes=True,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.3,
                 coords='centroids',
                 normalize_coords=False):
        predictor_sizes = np.array(predictor_sizes)
        if len(predictor_sizes.shape) == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != len(
                    predictor_sizes) + 1):
                raise ValueError(
                    "It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} "
                    "and len(predictor_sizes)+1 == {}".format(
                        len(scales), len(predictor_sizes) + 1))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError(
                    "All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        else:
            if not 0 < min_scale <= max_scale:
                raise ValueError(
                    "It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(
                        min_scale, max_scale))

        if aspect_ratios_per_layer:
            if (len(aspect_ratios_per_layer) != len(
                    predictor_sizes)):
                raise ValueError(
                    "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == "
                    "len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(
                        len(aspect_ratios_per_layer), len(predictor_sizes)))
            for aspect_ratios in aspect_ratios_per_layer:
                aspect_ratios = np.array(aspect_ratios)
                if np.any(aspect_ratios <= 0):
                    raise ValueError("All aspect ratios must be greater than zero.")
        else:
            if not aspect_ratios_global:
                raise ValueError(
                    "At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` cannot be `None`.")
            aspect_ratios_global = np.array(aspect_ratios_global)
            if np.any(aspect_ratios_global <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if neg_iou_threshold > pos_iou_threshold:
            raise ValueError("It cannot be `neg_iou_threshold > pos_iou_threshold`.")

        if not (coords == 'minmax' or coords == 'centroids'):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios_global = aspect_ratios_global
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.variances = variances
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.coords = coords
        self.normalize_coords = normalize_coords

        # Compute the number of boxes per cell
        if aspect_ratios_per_layer:
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

    def generate_anchor_boxes(self,
                              batch_size,
                              feature_map_size,
                              aspect_ratios,
                              this_scale,
                              next_scale,
                              diagnostics=False):
        '''
        Compute an array of the spatial positions and sizes of the anchor boxes for one particular classification
        layer of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            batch_size (int): The batch size.
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            diagnostics (bool, optional): If true, two additional outputs will be returned.
                1) An array containing `(width, height)` for each box aspect ratio.
                2) A tuple `(cell_height, cell_width)` meaning how far apart the box centroids are placed
                   vertically and horizontally.
                This information is useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.

        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        aspect_ratios = np.sort(aspect_ratios)
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        n_boxes = len(aspect_ratios)
        for ar in aspect_ratios:
            if (ar == 1) & self.two_boxes_for_ar1:
                # Compute the regular anchor box for aspect ratio 1 and...
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
                # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                w = np.sqrt(this_scale * next_scale) * size * np.sqrt(ar)
                h = np.sqrt(this_scale * next_scale) * size / np.sqrt(ar)
                wh_list.append((w, h))
                # Add 1 to `n_boxes` since we seem to have two boxes for aspect ratio 1
                n_boxes += 1
            else:
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
        wh_list = np.array(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios
        cell_height = self.img_height / feature_map_size[0]
        cell_width = self.img_width / feature_map_size[1]
        cx = np.linspace(cell_width / 2, self.img_width - cell_width / 2, feature_map_size[1])
        cy = np.linspace(cell_height / 2, self.img_height - cell_height / 2, feature_map_size[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2minmax')

        # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.limit_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 1]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 1]] = x_coords
            y_coords = boxes_tensor[:, :, :, [2, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [2, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, :2] /= self.img_width
            boxes_tensor[:, :, :, 2:] /= self.img_height

        if self.coords == 'centroids':
            # Convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='minmax2centroids')

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = np.tile(boxes_tensor, (batch_size, 1, 1, 1, 1))

        # Now reshape the 5D tensor above into a 3D tensor of shape
        # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
        # order of the tensor content will be identical to the order obtained from the reshaping operation
        # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
        # use the same default index order, which is C-like index ordering)
        boxes_tensor = np.reshape(boxes_tensor, (batch_size, -1, 4))

        if diagnostics:
            return boxes_tensor, wh_list, (int(cell_height), int(cell_width))
        else:
            return boxes_tensor

    def generate_encode_template(self, batch_size, diagnostics=False):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the conv net model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that `y_encoded`
        has this specific form.

        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documnentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 8)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 8` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes.
        '''

        # 1: Get the anchor box scaling factors for each conv layer from which we're going to make predictions
        #    If `scales` is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`
        if self.scales is None:
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes) + 1)

        # 2: For each conv predictor layer (i.e. for each scale factor) get the tensors for
        #    the anchor box coordinates of shape `(batch, n_boxes_total, 4)`
        boxes_tensor = []
        if diagnostics:
            wh_list = []  # List to hold the box widths and heights
            cell_sizes = []  # List to hold horizontal and vertical distances between any two boxes
            if self.aspect_ratios_per_layer:
                for i in range(len(self.predictor_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.predictor_sizes[i],
                                                                  aspect_ratios=self.aspect_ratios_per_layer[i],
                                                                  this_scale=self.scales[i],
                                                                  next_scale=self.scales[i + 1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
            else:  # Use the same global aspect ratio list for all layers
                for i in range(len(self.predictor_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.predictor_sizes[i],
                                                                  aspect_ratios=self.aspect_ratios_global,
                                                                  this_scale=self.scales[i],
                                                                  next_scale=self.scales[i + 1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
        else:
            if self.aspect_ratios_per_layer:
                for i in range(len(self.predictor_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.predictor_sizes[i],
                                                                   aspect_ratios=self.aspect_ratios_per_layer[i],
                                                                   this_scale=self.scales[i],
                                                                   next_scale=self.scales[i + 1],
                                                                   diagnostics=False))
            else:
                for i in range(len(self.predictor_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.predictor_sizes[i],
                                                                   aspect_ratios=self.aspect_ratios_global,
                                                                   this_scale=self.scales[i],
                                                                   next_scale=self.scales[i + 1],
                                                                   diagnostics=False))

        boxes_tensor = np.concatenate(boxes_tensor,
                                      axis=1)  # Concatenate the anchor tensors from the individual layers to one

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances  # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encode_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encode_template, wh_list, cell_sizes
        else:
            return y_encode_template

    def encode_y(self, ground_truth_labels):
        '''
        Convert ground truth bounding box data into a suitable format to train an SSD model.

        For each image in the batch, each ground truth bounding box belonging to that image will be compared against each
        anchor box in a template with respect to their jaccard similarity. If the jaccard similarity is greater than
        or equal to the set threshold, the boxes will be matched, meaning that the ground truth box coordinates and class
        will be written to the the specific position of the matched anchor box in the template.

        The class for all anchor boxes for which there was no match with any ground truth box will be set to the
        background class, except for those anchor boxes whose IoU similarity with any ground truth box is higher than
        the set negative threshold (see the `neg_iou_threshold` argument in `__init__()`).

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, xmax, ymin, ymax)`, and `class_id` must be an integer greater than 0 for all boxes
                as class_id 0 is reserved for the background class.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, and the last four elements are just dummy elements.
        '''

        # 1: Generate the template for y_encoded
        y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels), diagnostics=False)
        y_encoded = np.copy(y_encode_template)  # We'll write the ground truth box data to this array

        # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
        #    and for each matched box record the ground truth coordinates in `y_encoded`.
        #    Every time there is no match for a anchor box, record `class_id` 0 in `y_encoded` for that anchor box.

        class_vector = np.eye(self.n_classes)  # An identity matrix that we'll use as one-hot class vectors

        for i in range(y_encode_template.shape[0]):  # For each batch item...
            available_boxes = np.ones((y_encode_template.shape[
                                           1]))
            negative_boxes = np.ones((y_encode_template.shape[1]))  # 1 for all negative boxes, 0 otherwise
            for true_box in ground_truth_labels[i]:  # For each ground truth box belonging to the current batch item...
                true_box = true_box.astype(np.float)
                if abs(true_box[2] - true_box[1] < 0.001) or abs(true_box[4] - true_box[
                    3] < 0.001): continue
                if self.normalize_coords:
                    true_box[1:3] /= self.img_width  # Normalize xmin and xmax to be within [0,1]
                    true_box[3:5] /= self.img_height  # Normalize ymin and ymax to be within [0,1]
                if self.coords == 'centroids':
                    true_box = convert_coordinates(true_box, start_index=1, conversion='minmax2centroids')
                similarities = iou(y_encode_template[i, :, -12:-8], true_box[1:],
                                   coords=self.coords)  # The iou similarities for all anchor boxes
                negative_boxes[
                    similarities >= self.neg_iou_threshold] = 0
                similarities *= available_boxes
                available_and_thresh_met = np.copy(similarities)
                available_and_thresh_met[
                    available_and_thresh_met < self.pos_iou_threshold] = 0
                assign_indices = np.nonzero(available_and_thresh_met)[
                    0]  # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                if len(assign_indices) > 0:  # If we have any matches
                    y_encoded[i, assign_indices, :-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]),
                                                                       axis=0)
                    available_boxes[
                        assign_indices] = 0  # Make the assigned anchor boxes unavailable for the next ground truth box
                else:  # If we don't have any matches
                    best_match_index = np.argmax(
                        similarities)  # Get the index of the best iou match out of all available boxes
                    y_encoded[i, best_match_index, :-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]),
                                                                         axis=0)
                    available_boxes[
                        best_match_index] = 0  # Make the assigned anchor box unavailable for the next ground truth box
                    negative_boxes[best_match_index] = 0  # The assigned anchor box is no longer a negative box
            # Set the classes of all remaining available anchor boxes to class zero
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i, background_class_indices, 0] = 1

        # 3: Convert absolute box coordinates to offsets from the anchor boxes and normalize them
        if self.coords == 'centroids':
            y_encoded[:, :, [-12, -11]] -= y_encode_template[:, :,
                                           [-12, -11]]  # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:, :, [-12, -11]] /= y_encode_template[:, :, [-10, -9]] * y_encode_template[:, :, [-4,
                                                                                                         -3]]
            y_encoded[:, :, [-10, -9]] /= y_encode_template[:, :, [-10, -9]]  # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:, :, [-10, -9]] = np.log(y_encoded[:, :, [-10, -9]]) / y_encode_template[:, :, [-2,
                                                                                                       -1]]
        else:
            y_encoded[:, :, -12:-8] -= y_encode_template[:, :, -12:-8]  # (gt - anchor) for all four coordinates
            y_encoded[:, :, [-12, -11]] /= np.expand_dims(y_encode_template[:, :, -11] - y_encode_template[:, :, -12],
                                                          axis=-1)
            y_encoded[:, :, [-10, -9]] /= np.expand_dims(y_encode_template[:, :, -9] - y_encode_template[:, :, -10],
                                                         axis=-1)
            y_encoded[:, :, -12:-8] /= y_encode_template[:, :, -4:]

        return y_encoded
