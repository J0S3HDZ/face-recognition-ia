#Omar Muñoz Gatica
import cv2
import numpy as np
import imutils
import os

# se crea la carpeta si no existe para aguardar las imagenes del objeto a identificar
Datos = 'p'#nombre de la carpeta este se debe de cambiar al momento de ejecutar
if not os.path.exists(Datos):# si la carpeta no existe se crea
    print('Carpeta creada: ',Datos) #mandamos a imprimir que se acaba de crear la carpeta
    os.makedirs(Datos) #se crea la carpeta
cap = cv2.VideoCapture(2,cv2.CAP_DSHOW) #Se recolecta los datos con la camara

#Se crea x1,y1,x2,y2 para poder crear el retangudo donde estara el objeto
x1, y1 = 190, 80
x2, y2 = 450, 398
#el count nos premite guardar las imagenes tomadas
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    imAux = frame.copy()
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2) #Se genera el  -
    #retangulo para poder tener el objeto en cuadrado para tomar las fotos
    objeto = imAux[y1:y2,x1:x2] #se almacena el objeto
    objeto = imutils.resize(objeto,width=38) #le damos un tamaño a nuestra ventana
    #print(objeto.shape)
    k = cv2.waitKey(1)
    #con el if cada ves que precionemos la tecla "s" se va aguardar y nombrar la imagen que se va a tomar del retangulo
    if k == ord('s'):
        cv2.imwrite(Datos+'/objeto_{}.jpg'.format(count),objeto)
        print('Imagen guardada:'+'/objeto_{}.jpg'.format(count))
        count = count +1
    if k == 27:
        break
    cv2.imshow('frame',frame) #Se abre una nueva venta
    cv2.imshow('objeto',objeto) #se muestra lo que se encuentra en nuestro retangulo
cap.release()
cv2.destroyAllWindows()
