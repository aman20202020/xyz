
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Define a flask app
app = Flask(__name__)#, template_folder='/content/Untitled Folder')

# Model saved with Keras model.save()
MODEL_PATH ='malariaN.h5'

# Load your trained model
model = load_model(MODEL_PATH)

import cv2
   

#run_with_ngrok(app)
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index1.html')


@app.route("/prediction", methods=["POST"])
def prediction():

	img = request.files['img']

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (224,224))

	image = np.reshape(image, (1,224,224,3))

	pred = model.predict(image)

	pred = np.argmax(pred,axis=1)

	pred = s(pred)

  

	return render_template("prediction.html", data=pred)

def s(pred):
  if pred==0:
    pred="Parasite"
  elif pred==1:
    pred="not having malaria"
  else:
    pred="none"

  return pred      

if __name__ == '__main__':
    app.run(debug=True)