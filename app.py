import numpy as np
from flask import Flask, request, render_template, Response
import pickle
import cv2
import mediapipe as mp
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('squatmodel.pkl','rb'))

mpPose = mp.solutions.pose

@app.route('/')
def index():
    return render_template('Index_FITFAIZ.html')

@app.route("/video_page",methods = ['GET','POST'])
def vp():
    return render_template('blank.html')

@app.route("/page2",methods = ['GET','POST'])
def p2():
    return render_template('page2.html')

@app.route("/contact",methods = ['GET','POST'])
def CT():
    return render_template('contact.html')

@app.route("/SL",methods = ['GET','POST'])
def ht():
    return render_template('select.html')

def gen():
    cap = cv2.VideoCapture('1.mp4')
    mode = 0
    correct = 0
    count = 0
    direc = 0
    dcount = 30
    ncount = 3
    pcount = 1
    show = 0
    text = 'NEW'
    w=260

    while cap.isOpened():
        if show == 0:
            ret, frame = cap.read()
        if show == 1:
            frame = cv2.imread('exit.jpg')
            if count > 9:
                w=210
            if count > 99:
                w=150
            cv2.putText(frame, str(int(count)), (w,280), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8, cv2.LINE_AA)

        frame2 = cv2.resize(frame,(640,480))
        
        image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        results = mpPose.Pose().process(image)
    
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        try:
            poseX = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in poseX]).flatten())
            
            X = pd.DataFrame([pose_row])
            model_class = model.predict(X)[0]

            if model_class == 'exit':
                cv2.putText(image, str('STOP'), (240,210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
                pcount -= 1
                if pcount == -5:
                    show = 1

            cv2.rectangle(image,(150 ,0),(500,480),(0, 255, 0),3)

            if mode == 0:
                cv2.putText(image, str('READY?'), (210,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                dcount = 30
                if model_class == 'up':
                    mode = 1
            if mode == 1:
                dcount +=1
                cv2.putText(image, str(ncount), (275,250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)
                if dcount > 40:
                    mode = 2
            if mode == 2:
                dcount +=1
                cv2.putText(image, str(ncount-1), (275,250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)
                if dcount > 50:
                    mode = 3
            if mode == 3:
                dcount +=1
                cv2.putText(image, str(ncount-2), (275,250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)
                if dcount > 60:
                    mode = 4
            if mode == 4:
                cv2.putText(image, str('Squats 1 time'), (200,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str('Stay in your position'), (160,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if model_class == 'down':
                    if direc == 0:
                        pcount += 0.5
                        direc = 1
                if model_class == 'up':
                    if direc == 1:
                        pcount += 0.5
                        direc = 0
                if pcount == 2:
                    text = 'PERFECT'
                if text == 'PERFECT':    
                    cv2.putText(image, str(text), (190,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5, cv2.LINE_AA)
                    dcount +=1
                    if dcount > 65:
                        pcount -= 1
                        mode = 5
            if mode == 5:
                cv2.putText(image, str('Stay in your position'), (160,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if model_class == 'down':
                    if direc == 0:
                        count += 0.5
                        direc = 1
                        
                if model_class == 'up':
                    if direc == 1:
                        count += 0.5
                        direc = 0
                        
                cv2.rectangle(image, (0,0), (100, 90), (37, 117, 242), -1)

                cv2.putText(image, 'COUNT'
                    , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(int(count))
                    , (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

        except:
            pass    
        
        frame = cv2.imencode('.jpeg', image)[1].tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__=="__main__":
    app.run(debug=True)
