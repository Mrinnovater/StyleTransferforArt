from flask import Flask, render_template, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import pickle
import os
from utils.style_transfer_utils import load_image, style_transfer

# Cache TensorFlow Hub modules locally
os.environ['TFHUB_CACHE_DIR'] = './tfhub_cache'

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained style transfer model from TF Hub (fast version)
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    # Read uploaded files
    content_bytes = request.files['content'].read()
    style_bytes = request.files['style'].read()

    # Preprocess
    content_image = load_image(content_bytes)
    style_image = load_image(style_bytes)

    # Apply style transfer using TF Hub (optimized, instant)
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    # Convert tensor to image
    final_image = tf.squeeze(stylized_image) * 255
    final_image = tf.cast(final_image, tf.uint8).numpy()
    img = Image.fromarray(final_image)

    output_path = 'static/generated_img.jpg'
    img.save(output_path)

    # Save as .pkl
    with open('static/generated_img.pkl', 'wb') as file:
        pickle.dump(stylized_image.numpy(), file)

    return render_template('index.html', image_url='/static/generated_img.jpg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
