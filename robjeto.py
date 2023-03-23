import cv  # importamos libreria
# le decimos que vamos a usar la camara

cap = cv2.VideoCapture(0)  # dependiendo las camaras que tengas sera el numero puede ser 0,1,2,3........
# vamos a importar la libreria a traves de su ruta esto depende de donde lo tienes aguardado
# la ruta puede ser 'proyecto1/IA/cascade.xml' de pendiendo donde lo tengas esa será la ruta
majinBooClassif = cv2.CascadeClassifier('cascade.xml')
while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convestimos a escala de grises las imagenes
    toy = majinBooClassif.detectMultiScale(gray,
                                           scaleFactor=6,
                                           minNeighbors=25,
                                           minSize=(60, 68))
    for (x, y, w, h) in toy:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # se dibuja el retangulo
        cv2.putText(frame, 'Libro', (x, y - 10), 2, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)  # se le va a poner el nombre al objeto
    cv2.imshow('frame', frame)  # se abre una ventana nueva

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
