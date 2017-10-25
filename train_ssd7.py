from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator
import os

# global constants
img_height = 300
img_width = 480
img_channels = 3
n_classes = 6
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light']

scales = [0.08, 0.16, 0.32, 0.64, 0.96]
aspect_ratios = [0.5, 1.0, 2.0]
two_boxes_for_ar1 = True
limit_boxes = False
variances = [1.0, 1.0, 1.0, 1.0]
coords = 'centroids'
normalize_coords = False

min_scale = 0.08
max_scale = 0.96

model_file = './model/ssd7_model.h5'
model_weights_file = './model/ssd7_weights.h5'


def build_model():
    K.clear_session()
    model, predictor_sizes = ssd_300(image_size=(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=min_scale,
                                     max_scale=max_scale,
                                     scales=scales,
                                     aspect_ratios_global=None,
                                     aspect_ratios_per_layer=aspect_ratios,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     coords=coords,
                                     normalize_coords=normalize_coords)
    if os.path.exists(model_file):
        model.load_weights(model_weights_file)
    return model, predictor_sizes


def lr_schedule(epoch):
    if epoch <= 20:
        return 0.001
    else:
        return 0.0001


def train():
    batch_size = 32
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=0.1)

    model, predictor_sizes = build_model()
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    min_scale=min_scale,
                                    max_scale=max_scale,
                                    scales=scales,
                                    aspect_ratios_global=None,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    limit_boxes=limit_boxes,
                                    variances=variances,
                                    pos_iou_threshold=0.5,
                                    neg_iou_threshold=0.2,
                                    coords=coords,
                                    normalize_coords=normalize_coords)

    train_dataset = BatchGenerator(images_path='./data',
                                   include_classes='all',
                                   box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

    train_dataset.parse_csv(labels_path='./data/train_labels.csv',
                            input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             train=True,
                                             ssd_box_encoder=ssd_box_encoder,
                                             equalize=False,
                                             brightness=(0.5, 2, 0.5),
                                             flip=0.5,
                                             translate=((0, 30), (0, 30), 0.5),
                                             scale=(0.75, 1.2, 0.5),
                                             random_crop=(300, 300, 1, 3),
                                             crop=False,
                                             resize=False,
                                             gray=False,
                                             limit_boxes=True,
                                             include_thresh=0.4,
                                             diagnostics=False)

    n_train_samples = train_dataset.get_n_samples()
    epochs = 10
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=ceil(n_train_samples / batch_size),
                        epochs=epochs,
                        callbacks=[ModelCheckpoint('./model/ssd7_model_'
                                                   'epoch{epoch:02d}_loss{loss:.4f}.h5',
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='auto',
                                                   period=1),
                                   LearningRateScheduler(lr_schedule),
                                   EarlyStopping(monitor='val_loss',
                                                 min_delta=0.001,
                                                 patience=2),
                                   ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=0,
                                                     epsilon=0.001,
                                                     cooldown=0)])

    model.save(model_file)
    model.save_weights(model_weights_file)

    print()
    print("Model saved as {}.h5".format(model_file))
    print("Weights also saved separately as {}_weights.h5".format(model_weights_file))
    print()


def predict():
    model, _ = build_model()
    val_dataset = BatchGenerator(images_path='./Datasets/VOCdevkit/VOC2012/JPEGImages/',
                                 include_classes='all',
                                 box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

    val_dataset.parse_csv(labels_path='./data/val_labels.csv',
                          input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

    predict_generator = val_dataset.generate(batch_size=1,
                                             train=False,
                                             equalize=False,
                                             brightness=False,
                                             flip=False,
                                             translate=False,
                                             scale=False,
                                             random_crop=False,
                                             crop=False,
                                             resize=False,
                                             gray=False,
                                             limit_boxes=True,
                                             include_thresh=0.4,
                                             diagnostics=False)
    X, y_true, filenames = next(predict_generator)

    i = 0  # Which batch item to look at

    print("Image:", filenames[i])
    print()
    print("Ground truth boxes:\n")
    print(y_true[i])

    y_pred = model.predict(X)
    y_pred_decoded = decode_y2(y_pred,
                               confidence_thresh=0.5,
                               iou_threshold=0.4,
                               top_k='all',
                               input_coords='centroids',
                               normalize_coords=False,
                               img_height=None,
                               img_width=None)

    print("Decoded predictions (output format is [class_id, confidence, xmin, xmax, ymin, ymax]):\n")
    print(y_pred_decoded[i])

    plt.figure(figsize=(20, 12))
    plt.imshow(X[i])

    current_axis = plt.gca()
    # Draw the predicted boxes in blue
    for box in y_pred_decoded[i]:
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle((box[2], box[4]), box[3] - box[2], box[5] - box[4], color='blue', fill=False, linewidth=2))
        current_axis.text(box[2], box[4], label, size='x-large', color='white',
                          bbox={'facecolor': 'blue', 'alpha': 1.0})

    for box in y_true[i]:
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(
            plt.Rectangle((box[1], box[3]), box[2] - box[1], box[4] - box[3], color='green', fill=False, linewidth=2))
    plt.show()


if __name__ == '__main__':
    train()
