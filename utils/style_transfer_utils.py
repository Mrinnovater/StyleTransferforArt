import tensorflow as tf
import numpy as np
from PIL import Image
import io

def load_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

def style_transfer(content_image, style_image, model):
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image


