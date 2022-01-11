import cv2
import tensorflow 
from fer import FER
import matplotlib.pyplot as plt 


img1 = cv2.imread("happy.png")
print(img1.shape)
img1= cv2.rectangle(img1, (35, 10),(260, 290), (0, 255, 0), 4)
Detecteur= FER()
print(Detecteur.detect_emotions(img1))
Emotion = Detecteur.top_emotion(img1)
print("l'émotion détéctée sur l'image 1 est :", Emotion)
plt.imshow(img1)
plt.title('Happy')
plt.show()

img2 = cv2.imread("sad.png")
img2= cv2.rectangle(img2, (35, 10),(260, 290), (0, 255, 0), 4)
Detecteur= FER()
Emotion = Detecteur.top_emotion(img2)
print("l'émotion détéctée sur l'image 2 est :", Emotion)
plt.imshow(img2)
plt.title('Sad')
plt.show()

img3 = cv2.imread("surprise.png")
img3= cv2.rectangle(img3, (35, 10),(260, 290), (0, 255, 0), 4)
Detecteur= FER()
Emotion = Detecteur.top_emotion(img3)
print("l'émotion détéctée sur l'image 3 est :", Emotion)
plt.imshow(img3)
plt.title('Surprise')
plt.show()

img4 = cv2.imread("neutral.png")
img4= cv2.rectangle(img4, (35, 10),(260, 290), (0, 255, 0), 4)
Detecteur= FER()
Emotion = Detecteur.top_emotion(img4)
print("l'émotion détéctée sur l'image 4 est :", Emotion)
plt.imshow(img4)
plt.title('Neutral')
plt.show()

img5 = cv2.imread("angry.png")
img5= cv2.rectangle(img5, (35, 10),(260, 290), (0, 255, 0), 4)
Detecteur= FER()
Emotion = Detecteur.top_emotion(img5)
print("l'émotion détéctée sur l'image 5 est :", Emotion)
plt.imshow(img5)
plt.title('Angry')
plt.show()