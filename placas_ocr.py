import cv2
import numpy as np
import pytesseract
from PIL import Image

cap = cv2.VideoCapture(".\data\video\placas_ocr.mp4")

ctexto = ''

while True:

    ret, frame = cap.read()

    if ret == False:
        break

    cv2.rectangle(frame, (870,750), (1070, 850), (0,0, 0), cv2.FILLED)
    cv2.putText(frame,ctexto[0:9], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    al, an, c = frame.shape

    x1 = int(an /3)
    x2 = int(x1 * 2)

    y1 = int(al / 3)
    y2 = int(x1 * 2)

    cv2.rectangle(frame, (x1 + 100, y1 +500), (1120, 940), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    recorte = frame[y1:y2, x1:x2]

    nB = np.matrix(recorte[:, :, 0])
    nG = np.matrix(recorte[:, :, 1])
    nR = np.matrix(recorte[:, :, 2])


    color = cv2.absdiff(nG, nB)

    _, umbral = cv2.threshold(color, 40, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contornos = sorted(contornos, key = lambda x: cv2.contourArea(x), reverse=True)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500 and area < 5000:
            x, y, ancho, alto = cv2.boundingRect(contorno)

            xpi = x + x1
            ypi = y + y1

            xpf = x + ancho + x1
            ypf = y + alto + y1

            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)

            placa = frame[ypi : ypf, xpi : xpf]

            alp, anp, cp = placa.shape

            mva = np.zeros((alp, anp))

            nBp = np.matrix(recorte[:, :, 0])
            nGp = np.matrix(recorte[:, :, 1])
            nRp = np.matrix(recorte[:, :, 2])

            for col in range(0, alp):
                for fil in range(0, anp):
                    maximo = max(nRp[col, fil], nGp[col, fil], nBp[col, fil],)
                    mva[col, fil] = 255 - maximo

            
            _, bin = cv2.threshold(mva, 150, 255, cv2.THRESH_BINARY)

            bin = bin.reshape(alp, anp)
            bin = Image.fromarray(bin)
            bin = bin.convert('L')

            if alp >= 36 and anp >= 82:
                pytesseract.pytesseract.tesseract_cmd = r'C:\Programs Files\Tesseract-OCR\tesseract.exe'
                config = "--psm 1"
                texto = pytesseract.image_to_string(bin, config=config)

                if len(texto) >=9:
                    ctexto = texto

            break

    cv2.inshow("Vehiculos", frame)

    t = cv2.waitKey(1)

    if  t == 27:
        break
cap.release()
cv2.destroyAllWindows()
