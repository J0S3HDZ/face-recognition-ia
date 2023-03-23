import cv2
import os
import imutils

personName = 'jose'
dataPath = 'C:/Users/Jose Hernandez/PycharmProjects/proyectoIA/Data'  #Ubicacion de guardado de caras
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):  #Si la carpeta con el nombre de la persona no existe
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)
print("Carpeta creada...")
print("Camara abierta...")
cap = cv2.VideoCapture(0) #Abre webcam
print("Paso de camara...")
print("Abrir haarcascade...")
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #Abre haarcasscade para reconocimiento
print("Paso de haarcascade...")
count = 0  #contador para maximo de imagenes

while True:     #Ciclo para capturar cada frame
    ret,frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame,width=640)  #Cambia tamaño de cara capturada
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Covercion a escala de grises
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5) #Deteccion de caras mediante la escala de grises

    for (x, y, w, h) in faces:      #Dibujo de recuadro al momento de reconocer la cara
        cv2.rectangle(frame,(x,y),(x+w,y+h),(128,0,255),2)      #Rectangulo
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)    #Tamaño del guardado de cara
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)     #Formato del nombre de cada imagen

        count = count + 1 #Aumento de count
        cv2.imshow('frame',frame)  #Abrir ventana

    k = cv2.waitKey(1) #Cierra cuando das ESC
    if k == 27 or count >= 100:     #Si guarda 100 imagenes finalizar
        break

cap.release()  #Cierra webcam
cv2.destroyAllWindows()     #Cierra ventanas abiertas
