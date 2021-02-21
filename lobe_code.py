import cv2 
from tf_example import Model
from PIL import Image

PATH = 'Mask Detection TensorFlow'

model = Model(PATH) # создаем экземпляр класса модели, загружаем модель
model.load()

cam = cv2.VideoCapture(0) # создаем экземпляр камеры с индексом 0

frame_counter = 0 
label = '?'

while True:
    _, frame = cam.read() # считываем значение с камеры

    frame = cv2.resize(frame, (640, 360))

    frame_counter += 1

    if frame_counter % 10 == 0:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # ковертируем BGR цвет в RGB, чтобы отправить в PIL 
        img = Image.fromarray(img) 

        label = model.predict(img)['Prediction'] # предсказываем значение Mask/No mask
        
        frame_counter = 0 # обнуляем значение счетчика кадров
        print(label) 
    
    
    cv2.putText(frame, label, (320, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2) # отрисовываем предсказанный класс на экран
    cv2.imshow('frame', frame) # выводим на экран окно с изображением 


    if cv2.waitKey(1) == 27:
        break


cam.release() # "освобождаем" камеру"
cv2.destroyAllWindows() # "удаляем все окна "
