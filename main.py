# Tira mensagens de alerta e informativos do tensorflow na hora da exec do cod:
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#importação das bibliotecas:
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

#Começo do Codigo/função:
def verify(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    #Deixando as imgs na cor certa
    plt.imshow(img1[:, :, ::-1])
    plt.imshow(img2[:, :, ::-1])

    resultado = DeepFace.verify(img1_path, img2_path)
    
    verificacao = resultado['verified']
    time_taken = resultado['time']
    
    print("Mesma Rosto: ", verificacao)
    print("Tempo de Verificação: ", time_taken)
    
    if verificacao:
        print("São os mesmos na fotos")
    else:
        print("Não são os mesmos na fotos!")

#Chamando a função e passando os rostos:
verify("rostos_img/bh.jpg", "rostos_img/gab2.jpeg")