

from flask import Flask, render_template, request, send_file
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import os
from utils.style_transfer_utils import load_image

# ⚙️ Configuration
USE_DISK = False  # Set True for local testing with static image file
TFHUB_CACHE_DIR = './tfhub_cache'
os.environ['TFHUB_CACHE_DIR'] = TFHUB_CACHE_DIR

# 🔧 Initialize Flask app
app = Flask(__name__)

# 📦 Load the style transfer model from TensorFlow Hub
print("🔄 Loading TensorFlow Hub model...")
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("✅ Model loaded successfully!")

# 🚀 TensorFlow function wrapper for performance
@tf.function
def stylize(content_image, style_image):
    return hub_model(content_image, style_image)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # 📤 Read uploaded files
        content_bytes = request.files['content'].read()
        style_bytes = request.files['style'].read()

        # 🖼️ Preprocess images
        content_image = load_image(content_bytes)
        style_image = load_image(style_bytes)

        # 🧠 Run style transfer
        stylized_image = stylize(content_image, style_image)

        # 🎨 Convert output to image format
        final_image = tf.squeeze(stylized_image) * 255
        final_image = tf.cast(final_image, tf.uint8).numpy()
        pil_image = Image.fromarray(final_image)

        if USE_DISK:
            # 💾 Save to disk (for local viewing)
            output_path = 'static/generated_img.jpg'
            if not os.path.exists('static'):
                os.makedirs('static')
            pil_image.save(output_path)
            return render_template('index.html', image_url='/' + output_path)
        else:
            # 🚀 Send image in-memory for cloud platforms
            img_io = io.BytesIO()
            pil_image.save(img_io, format='JPEG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return "An error occurred during image processing.", 500

# 🧪 Run locally
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
