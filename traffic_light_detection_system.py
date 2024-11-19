# prompt: i dont want keras_cv

import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import requests
import zipfile
from tqdm.auto import tqdm
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches


!pip install matplotlib
!pip install pycocotools
!pip uninstall opencv-python -y
!pip uninstall opencv-python-headless -y
!pip install opencv-python-headless
!pip install keras-core
!pip install matplotlib.py

# ## Imports

# from keras_cv import bounding_box
# from keras_cv import visualization
# ## Download Dataset
# Download dataset.
def download_file(url, save_name):
    if not os.path.exists(save_name):
        print(f"Downloading file")
        file = requests.get(url, stream=True)
        total_size = int(file.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True
        )
        with open(os.path.join(save_name), 'wb') as f:
            for data in file.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
    else:
        print('File already present')

download_file(
    'https://www.dropbox.com/scl/fi/suext2oyjxa0v4p78bj3o/S2TLD_720x1280.zip?rlkey=iequuynn54uib0uhsc7eqfci4&dl=1',
    'S2TLD_720x1280.zip'
)
# Unzip the data file
def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("./")
            print("Extracted all")
    except:
        print("Invalid file")

unzip('S2TLD_720x1280.zip')
# ## Dataset and Training Parameters
SPLIT_RATIO = 0.2
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCH = 75
GLOBAL_CLIPNORM = 10.0
IMG_SIZE = (640, 640)
# ## Dataset Preparation
class_ids = [
    "red",
    "yellow",
    "green",
    "off",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Path to images and annotations
path_images = "S2TLD_720x1280/images/"
path_annot = "S2TLD_720x1280/annotations/"

# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
    ]
)

# Get all JPEG image file paths in path_images and sort them
jpg_files = sorted(
    [
        os.path.join(path_images, file_name)
        for file_name in os.listdir(path_images)
        if file_name.endswith(".jpg")
    ]
)
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids


image_paths = []
bbox = []
classes = []
for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)
bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
# Determine the number of validation samples
num_val = int(len(xml_files) * SPLIT_RATIO)

# Split the dataset into train and validation sets
val_data = data.take(num_val)
train_data = data.skip(num_val)
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

augmenter = keras.Sequential(
    layers=[
        keras.layers.RandomFlip(mode="horizontal"),
        keras.layers.Resizing(height=IMG_SIZE[0], width=IMG_SIZE[1])
    ]
)
train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)



def random_flip_bounding_boxes(inputs, training):
  """Applies random horizontal flip to images and bounding boxes.

  Args:
    inputs: A dictionary containing 'images' and 'bounding_boxes'.
    training: A boolean indicating whether the model is in training mode.

  Returns:
    A dictionary with the flipped images and bounding boxes.
  """
  if not training:
      return inputs

  images = inputs['images']
  bounding_boxes = inputs['bounding_boxes']

  # Flip images horizontally
  flipped_images = tf.image.flip_left_right(images)

  # Get bounding box information
  boxes = bounding_boxes['boxes']
  classes = bounding_boxes['classes']

  # Calculate image width
  image_width = tf.cast(tf.shape(images)[2], tf.float32)

  # Flip the bounding box coordinates
  flipped_boxes = tf.stack([
      image_width - boxes[..., 2],  # xmin
      boxes[..., 1],               # ymin
      image_width - boxes[..., 0],  # xmax
      boxes[..., 3]                # ymax
  ], axis=-1)

  return {
      'images': flipped_images,
      'bounding_boxes': {
          'classes': classes,
          'boxes': flipped_boxes
      }
  }

resizing = keras.layers.Resizing(height=IMG_SIZE[0], width=IMG_SIZE[1])

val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(lambda x: {**x, 'images': resizing(x['images'])}, num_parallel_calls=tf.data.AUTOTUNE)



def visualize_dataset(dataset, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    for i, sample in enumerate(dataset.take(rows * cols)):
        ax = axes[i // cols, i % cols]
        image = sample['images'][0].numpy()  # Assuming batch dimension
        boxes = sample['bounding_boxes']['boxes'][0].numpy()  # Assuming batch dimension
        classes = sample['bounding_boxes']['classes'][0].numpy()

        # Scale image if necessary
        if value_range == (0, 255):
            image = image / 255.0
        ax.imshow(image)

        # Draw bounding boxes
        for box, cls in zip(boxes, classes):
            if bounding_box_format == "xyxy":
                xmin, ymin, xmax, ymax = box
                width, height = xmax - xmin, ymax - ymin
            else:
                raise ValueError(f"Unsupported bounding_box_format: {bounding_box_format}")

            rect = patches.Rectangle(
                (xmin, ymin), width, height, linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin, class_mapping[int(cls)], color="blue", fontsize=10)

        ax.axis("off")
    plt.tight_layout()
    plt.show()


visualize_dataset(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)
visualize_dataset(
    val_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)



