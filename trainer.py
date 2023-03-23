import cv2
import os
import numpy as np

dataPath = 'Data/'  #Ubicacion de guardado de caras
peopleList = os.listdir(dataPath)                                       #Paso de listado de carpetas
print('Lista de personas: ',peopleList)                                 #Muestra los nombres de las carpetas
labels = []                                                             #arreglo
facesData = []                                                          #arreglo
label = 0                                                               #variable

for nameDir in peopleList:                                              #Ciclo para leer todas las imagenes
    personPath = dataPath + '/' +nameDir
    print('Leyendo imagenes...')

    for fileName in os.listdir(personPath):                             #Recorrido de rostros para verlos
        print('Rostros: ',nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        cv2.imshow('image',image)
        cv2.waitKey(10)

face_recognizer = cv2.face.EigenFaceRecognizer_create()                 #Creacion del haarcascade para reconocer rostro

print('Entrenando...')
face_recognizer.train(facesData,np.array(labels))                       #Entrenamiento de la IA

#Almacenar el modelo obtenido
face_recognizer.write('modeloEigenFace.xml')                            #Creacion de la IA entrenada
print('Modelo almacenado :)')
