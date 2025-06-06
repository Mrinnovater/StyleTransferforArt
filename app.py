from flask import Flask, render_template, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import pickle
import os
from utils.style_transfer_utils import load_image

# Set local TF Hub cache
os.environ['TFHUB_CACHE_DIR'] = './tfhub_cache'

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained style transfer model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Optimize model call with tf.function
@tf.function
def stylize(content, style):
    return hub_model(content, style)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    content_bytes = request.files['content'].read()
    style_bytes = request.files['style'].read()

    content_image = load_image(content_bytes)
    style_image = load_image(style_bytes)

    stylized_image = stylize(content_image, style_image)

    # Convert and save as image
    output_path = 'static/generated_img.jpg'
    final_image = tf.squeeze(stylized_image) * 255
    final_image = tf.cast(final_image, tf.uint8).numpy()
    Image.fromarray(final_image).save(output_path)

    # Optional: Save numpy result
    with open('static/generated_img.pkl', 'wb') as f:
        pickle.dump(stylized_image.numpy(), f)

    return render_template('index.html', image_url='/' + output_path)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=5000)
