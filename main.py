#importação das bibliotecas:
from deepface import DeepFace
import os
import logging

# Tira mensagens de alerta e informativos do tensorflow na hora da exec do cod:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#Começo do Codigo/função:
def verify(img1_path, img2_path):
    resultado = DeepFace.verify(img1_path, img2_path)
    
    verificacao = resultado['verified']
    time_taken = resultado['time']
    
    print("Mesmo Rosto: ", verificacao)
    print("Tempo de Verificação: ", time_taken)
    
    if verificacao:
        print("Carimbado =)")
    else:
        print("Incapaz =(")

#Chamando a função e passando os rostos:
verify("rostos_img/foto-padrao.jpeg", "rostos_img/foto-escura.jpeg")