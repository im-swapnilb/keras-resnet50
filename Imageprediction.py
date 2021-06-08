from flask import Flask, request, jsonify, url_for, render_template
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


ALLOWED_EXTENSION = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])



def allowed_file(filename):
    return '.' in filename and \
     filename.rsplit('.',1)[1] in ALLOWED_EXTENSION



app = Flask(__name__)

model = ResNet50(weights='imagenet')

@app.route('/')

def index():
    return render_template('ImageML.html')



@app.route('/', methods=['POST'])

def upload_image():
    
    if 'image' not in request.files:
        
        return render_template('ImageML.html', prediction='No image found')
    
    file = request.files['image']
    
    if file.filename =='':
        
        return render_template('ImageML.html', prediction='Selected NO Image')
    
    if file and allowed_file(file.filename):
        
        filename_uploaded = secure_filename(file.filename)
        
        print("*"+filename_uploaded)
        path = "static/UPLOAD_FOLDER"
        file.save(os.path.join(path, secure_filename(file.filename)))
        img_path = path + "/" + filename_uploaded
        print(img_path)

        ''' Image pre-processing '''
        
        x = []
        

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        result = decode_predictions(preds, top=3)[0]
        print('Predicted:', decode_predictions(preds, top=3)[0])
        print("picture : ", result[0][1])
        print("Prob : ", result[0][2])
        picture_pred = result[0][1]
        probability = round(result[0][2]*100,2)

        
        response = {'pred': result}
        os.remove(img_path)
        return render_template('ImageML.html', prediction = 'The picture which you have uploaded can be {} with my prediction percentage {}'.format(picture_pred,probability))
    else:
        return render_template('ImageML.html', prediction = 'Invalid File extension')
    
if __name__=='__main__':
    
    app.run(debug=True, use_reloader=False)
