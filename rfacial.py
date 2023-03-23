import cv2
import os


dataPath = 'C:/Users/Jose Hernandez/PycharmProjects/proyectoIA/Data'  #Ubicacion de guardado de caras
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

#Se llaman las variables creadas anteriormente
face_recognizer = cv2.face.EigenFaceRecognizer_create()

face_recognizer.read('modeloEigenFace.xml')

cap = cv2.VideoCapture(0) #Abrir webcam

faceClassif =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Leemos el haarcascade

#Ciclo para ver cada frame de la camara y reconocer
while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Entrada de camara a escala de grises
    auxFrame = gray.copy()                         #copia de cada frame

    faces = faceClassif.detectMultiScale(gray,1.3,5) #Deteccion mediante escala de grises

    for (x, y, w, h) in faces:  #Ciclo para dibujar el cuadro al momento de reconocer al sujeto
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)      #Tamaño del recuadro alrededor del rostro
        result = face_recognizer.predict(rostro)            #IA para predecir quien es

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(0,255,0),1,cv2.LINE_AA) #Mostramos el nombre del sujeto reconocido

        if result[1] < 5700:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        else:   #Si la IA no reconoce al sujeto lo marca como desconocido
            cv2.putText(frame, 'Desconocido', (x, y - 25), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)



    cv2.imshow('frame',frame)  #Mostrar ventana
    k = cv2.waitKey(1)  #ESC para salir
    if k == 27:
        break

cap.release()   #Cierra la webcam
cv2.destroyAllWindows() #Cierra todas las ventanas


