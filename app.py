from flask import Flask , request
# from flask_cors import CORS, cross_origin
from fastai.data.all import *
import base64
import PIL.Image as Image
import io
from predictor import predict

app = Flask(__name__)
# p = Prediction()

@app.route('/t', methods=['POST'])
def hello_world():
    # fre = request.files['file'] 
    # print(request.form['file'])
    request.files
    image_read = request.form['file']

    # assumes first 8 chars are base64, TODO: fixt the assumption
    image_read = image_read[7:]
    # print(image_read[0:10])

    base64_img_bytes = image_read.encode('utf-8')
    b = base64.b64decode(base64_img_bytes)
    img = Image.open(io.BytesIO(b))
    img.save('rate.png')
    response = predict(r'./rate.png')
    # img.show()
    # print(response)
    # image_64_decode.save('.')s
    # image_result.save('.')
    # print(request.files)
    # file.save('.')
    print(response[0][0])

    return str(response[0][0])



