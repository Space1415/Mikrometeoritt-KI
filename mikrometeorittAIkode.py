import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from PIL import Image

app = Flask(__name__)

# Full sti til den tidligere trente modellen
model_path = r'C:\Users\adriva001\Downloads\AI\Mikrometeoritt_AI_model.keras'
model = tf.keras.models.load_model(model_path)

# Klassenavn for den nye modellen
class_names = ["Mikrometeoritt", "Ikke Mikrometeoritt"]

# Full sti for opplastingsmappe
UPLOAD_FOLDER = r'C:\Users\adriva001\Downloads\AI\uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hjelpefunksjon for å forberede bildet før prediksjon
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((160, 160))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img) / 255.0

    if img.shape != (160, 160, 3):
        raise ValueError(f"Uventet bildeform: {img.shape}. Forventet (160, 160, 3)")

    img = img.reshape((1, 160, 160, 3))
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("Ingen fil i forespørselen")
            return render_template('index1.html', error='Ingen filopplasting')

        file = request.files['file']

        if file.filename == '':
            print("Ingen fil valgt")
            return render_template('index1.html', error='Ingen fil valgt')

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"Fil lagret: {file_path}")

            try:
                img = preprocess_image(file_path)
                prediction = model.predict(img)
                probabilities = prediction[0]
                formatted_probabilities = ["{}: {:.2f}%".format(class_name, prob * 100) for class_name, prob in zip(class_names, probabilities)]
                print(f"Prediksjon: {formatted_probabilities}")
            except Exception as e:
                print(f"Feil under bildebehandling: {e}")
                return render_template('index1.html', error=str(e))

            return render_template('index1.html', prediction=formatted_probabilities)

    return render_template('index1.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=25565, debug=True)
