from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image, ImageEnhance
import random

app = Flask(__name__)

# Load the trained model
model = load_model("models/model.h5")

train_dir = 'archive/Training'
# Class labels
class_labels = sorted(os.listdir(train_dir))

# Define the uploads folder
UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)

# Set the upload folder configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Correct way to set config

def augment_image(image):
    # Convert to array first
    image = img_to_array(image)
    # Convert to PIL Image for enhancement
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    # Convert back to array and normalize
    image = np.array(image)/255.0
    return image

def predict_tumor(image_path, model):
    # Load and preprocess image exactly as in training
    img = load_img(image_path, target_size=(128, 128))
    img = augment_image(img)  # Use the same augment_image function from training
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img, axis=0)
    
    # Prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_index]
    
    # Determine the class
    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

@app.route("/", methods=['GET', 'POST'])
def index():
  if request.method=='POST':
    # Handle file upload
    file = request.files['file']
    if file:
      # save the file
      file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  # Use square brackets
      file.save(file_location)

      # predict results
      result, confidence = predict_tumor(file_location, model)

      return render_template('index.html', result=result, confidence=f'{confidence*100:.2f}%', file_path=f'/uploads/{file.filename}')
  return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
   return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__=='__main__':
   app.run(debug=True)