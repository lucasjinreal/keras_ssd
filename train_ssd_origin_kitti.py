import os
from math import ceil

from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from ssd.keras_ssd300 import ssd_300
from ssd.keras_ssd_loss import SSDLoss
from ssd.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y

from ssd.ssd_batch_generator import BatchGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# global constants
img_height = 1080
img_width = 375
img_channels = 3
n_classes = 9
classes = ['Person_sitting', 'Car', 'DontCare', 'Pedestrian',
           'Misc', 'Truck', 'Cyclist', 'Tram', 'Van']

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
model_file = os.path.join(model_save_dir, 'kitti_ssd_origin_model.h5')
model_weights_file = os.path.join(model_save_dir, 'kitti_ssd_origin_weights.h5')

image_dir = '/media/jintian/Netac/Datasets/Kitti/object/training/image_2'
label_file = './data/kitti/kitti_simple_label.txt'


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
    if os.path.exists(model_weights_file):
        print('load from previous weights: ', model_weights_file)
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

    train_dataset = BatchGenerator(images_path=image_dir,
                                   include_classes='all',
                                   box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

    train_dataset.parse_txt(image_dir, label_file)

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             train=True,
                                             ssd_box_encoder=ssd_box_encoder,
                                             equalize=False,
                                             brightness=(0.5, 2, 0.5),
                                             flip=0.5,
                                             translate=((0, 30), (0, 30), 0.5),
                                             scale=(0.75, 1.2, 0.5),
                                             random_crop=(img_height, img_width, 1, img_channels),
                                             crop=False,
                                             resize=False,
                                             gray=False,
                                             limit_boxes=True,
                                             include_thresh=0.4,
                                             diagnostics=False)

    n_train_samples = train_dataset.get_n_samples()
    epochs = 50

    try:
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=ceil(n_train_samples / batch_size),
                            epochs=epochs,
                            callbacks=[LearningRateScheduler(lr_schedule)])
    except KeyboardInterrupt:
        model.save(model_file)
        model.save_weights(model_weights_file)
    model.save(model_file)
    model.save_weights(model_weights_file)

    print()
    print("Model saved as {}".format(model_file))
    print("Weights also saved separately as {}".format(model_weights_file))
    print()


def load_image_from_file(img_file, target_size):
    if os.path.exists(img_file):
        with Image.open(img_file) as img:
            img_array = np.asarray(img)
        img_shape = img_array.shape
        assert target_size[0] <= img_shape[0] and target_size[1] <= img_shape[1], 'image file must bigger than ' \
                                                                                  'target_size'

        crop_w = np.random.randint(0, img_array.shape[0] - target_size[0] + 1)
        crop_h = np.random.randint(0, img_array.shape[1] - target_size[1] + 1)
        img_random_cropped = img_array[crop_w:crop_w + target_size[0], crop_h:crop_h + target_size[1]]
        img_random_cropped = np.expand_dims(img_random_cropped, 0)
        return img_random_cropped, img_array
    else:
        print('{} doest not exist.'.format(img_file))


def predict():
    model, _ = build_model()
    model.load_weights(model_weights_file)
    print('load model succeed.')

    image, origin_image = load_image_from_file('test_images/000123.png', target_size=(img_height, img_width,
                                                                                      img_channels))
    y_pred = model.predict(np.expand_dims(origin_image, 0))
    y_pred_decoded = decode_y(y_pred,
                              confidence_thresh=0.01,
                              iou_threshold=0.55,
                              top_k=200,
                              input_coords='centroids',
                              normalize_coords=normalize_coords,
                              img_height=img_height,
                              img_width=img_width)

    print("Predicted boxes:\n")
    print(y_pred_decoded)

    plt.figure(figsize=(16, 10))
    plt.imshow(origin_image)

    current_axis = plt.gca()
    # Draw the predicted boxes in blue
    for box in y_pred_decoded[0]:
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle((box[2], box[4]), box[3] - box[2], box[5] - box[4], color='blue', fill=False, linewidth=2))
        current_axis.text(box[2], box[4], label, size='x-large', color='white',
                          bbox={'facecolor': 'blue', 'alpha': 1.0})
    plt.show()


if __name__ == '__main__':
    train()
    # predict()