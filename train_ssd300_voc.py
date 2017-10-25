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
img_width = 300
img_channels = 3
n_classes = 21
classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
          1.05]
aspect_ratios = [[0.5, 1.0, 2.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0]]
two_boxes_for_ar1 = True
limit_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True

model_save_dir = './model'
model_file = os.path.join(model_save_dir, 'ssd300_model.h5')
model_weights_file = os.path.join(model_save_dir, 'ssd300_weights.h5')


def build_model():
    K.clear_session()
    model, predictor_sizes = ssd_300(image_size=(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=None,
                                     max_scale=None,
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
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    batch_size = 16

    model, predictor_sizes = build_model()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=0.1)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    min_scale=None,
                                    max_scale=None,
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

    train_dataset = BatchGenerator(images_path='./data/VOCdevkit/VOC2012/JPEGImages/',
                                   include_classes='all',
                                   box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

    train_dataset.parse_xml(annotations_path='./data/VOCdevkit/VOC2012/Annotations/',
                            image_set_path='./data/VOCdevkit/VOC2012/ImageSets/Main/',
                            image_set='train.txt',
                            classes=classes,
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

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

    val_dataset = BatchGenerator(images_path='./data/VOCdevkit/VOC2012/JPEGImages/',
                                 include_classes='all',
                                 box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

    val_dataset.parse_xml(annotations_path='./data/VOCdevkit/VOC2012/Annotations/',
                          image_set_path='./data/VOCdevkit/VOC2012/ImageSets/Main/',
                          image_set='val.txt',
                          classes=classes,
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         random_crop=(300, 300, 1, 3),
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4,
                                         diagnostics=False)

    n_val_samples = val_dataset.get_n_samples()

    epochs = 10

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=ceil(n_train_samples / batch_size),
                                  epochs=epochs,
                                  callbacks=[ModelCheckpoint('./model/ssd300_model_'
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
                                                           patience=2)],
                                  validation_data=val_generator,
                                  validation_steps=ceil(n_val_samples / batch_size))

    model.save(model_file)
    model.save_weights(model_weights_file)

    print()
    print("Model saved as {}.h5".format(model_file))
    print("Weights also saved separately as {}_weights.h5".format(model_weights_file))
    print()


def predict():
    model, _ = build_model()
    model.load_weights(model_weights_file)
    val_dataset = BatchGenerator(images_path='./data/VOCdevkit/VOC2012/JPEGImages/',
                                 include_classes='all',
                                 box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

    val_dataset.parse_xml(annotations_path='./data/VOCdevkit/VOC2012/Annotations/',
                          image_set_path='./data/VOCdevkit/VOC2012/ImageSets/Main/',
                          image_set='val.txt',
                          classes=classes,
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)
    predict_generator = val_dataset.generate(batch_size=1,
                                             train=False,
                                             equalize=False,
                                             brightness=False,
                                             flip=False,
                                             translate=False,
                                             scale=False,
                                             random_crop=(300, 300, 1, 3),
                                             crop=False,
                                             resize=False,
                                             gray=False,
                                             limit_boxes=True,
                                             include_thresh=0.4,
                                             diagnostics=False)
    X, y_true, file_names = next(predict_generator)

    i = 0  # Which batch item to look at

    print("Image:", file_names[i])
    print()
    print("Ground truth boxes:\n")
    print(y_true[i])
    y_pred = model.predict(X)
    y_pred_decoded = decode_y(y_pred,
                              confidence_thresh=0.01,
                              iou_threshold=0.45,
                              top_k=200,
                              input_coords='centroids',
                              normalize_coords=normalize_coords,
                              img_height=img_height,
                              img_width=img_width)

    print("Predicted boxes:\n")
    print(y_pred_decoded[i])


if __name__ == '__main__':
    train()
