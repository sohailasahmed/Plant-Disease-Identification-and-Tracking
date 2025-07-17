from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import geoip2.database
from keras.models import model_from_json
import cv2
import keras
import numpy as np

global load_model
global loaded_model
load_model = 0

plants = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Tomato_Bacterial_spot',
          'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
          'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

fertilizers = []
with open("messages.txt", "r") as file:
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        fertilizers.append(line)
file.close()

def getFertilizer(name):
    details = "Fertilizer Details Not Available"
    for i in range(len(fertilizers)):
        arr = fertilizers[i].split(":")
        arr[0] = arr[0].strip()
        arr[1] = arr[1].strip()
        if arr[0] == name:
            details = arr[1]
            break
    return details     

def Upload(request):
    if request.method == 'GET':
       return render(request, 'Upload.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def getClientIP(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip    

def Signup(request):
    if request.method == 'POST':
      #user_ip = getClientIP(request)
      #reader = geoip2.database.Reader('C:/Python/PlantDisease/GeoLite2-City.mmdb')
      #response = reader.city('103.48.68.11')
      #print(user_ip)
      #print(response.location.latitude)
      #print(response.location.longitude)
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      contact = request.POST.get('contact', False)
      email = request.POST.get('email', False)
      address = request.POST.get('address', False)
      
      db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'PlantDiseaseDB',charset='utf8')
      db_cursor = db_connection.cursor()
      student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
      db_cursor.execute(student_sql_query)
      db_connection.commit()
      print(db_cursor.rowcount, "Record Inserted")
      if db_cursor.rowcount == 1:
       context= {'data':'Signup Process Completed'}
       return render(request, 'Register.html', context)
      else:
       context= {'data':'Error in signup process'}
       return render(request, 'Register.html', context)    
        
def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        utype = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'PlantDiseaseDB',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    utype = 'success'
                    break
        if utype == 'success':
            file = open('session.txt','w')
            file.write(username)
            file.close()
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        if utype == 'none':
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)


def UploadImage(request):
    if request.method == 'POST':
        global load_model
        global loaded_model
        myfile = request.FILES['t1']
        fname = request.FILES['t1'].name
        fs = FileSystemStorage()
        if os.path.exists('PlantDiseaseApp/static/plant/test.png'):
            os.remove('PlantDiseaseApp/static/plant/test.png')
        filename = fs.save('PlantDiseaseApp/static/plant/test.png', myfile)

        
        user = ''
        with open("session.txt", "r") as file:
          for line in file:
              user = line.strip('\n')
        if load_model == 0:
            with open('model/model.json', "r") as json_file:
                loaded_model_json = json_file.read()
                loaded_model = model_from_json(loaded_model_json)

            loaded_model.load_weights("model/model_weights.h5")
            loaded_model._make_predict_function()   
            print(loaded_model.summary())
            load_model = 1

        img = cv2.imread('PlantDiseaseApp/static/plant/test.png')
        img = cv2.resize(img, (64,64))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,64,64,3)
        X = np.asarray(im2arr)
        X = X.astype('float32')
        X = X/255
        preds = loaded_model.predict(X)
        print(str(preds)+" "+str(np.argmax(preds)))
        predict = np.argmax(preds)
        print(plants[predict])
        img = im2arr.reshape(64,64,3)
        details = getFertilizer(plants[predict])

        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'PlantDiseaseDB',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO locations(username,image_name,predicted_disease) VALUES('"+user+"','"+fname+"','"+plants[predict]+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()

        
        html = ''
        html += '<font size="4" color="red"><center>Plant Condition Predicted as: <b>' + plants[predict] + '</b></center></font><br/><br/>'
        html += '<font size="4" color="green"><center>Recommended Solution: ' + details + '</center></font><br/><br/>'

        img = cv2.resize(img,(650,450))
        cv2.putText(img, 'Plant Condition Predicted as '+plants[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Plant Condition Predicted as '+plants[predict],img)
        cv2.waitKey(0)
        context= {'data':html}
        return render(request, 'Upload.html', context)

            
