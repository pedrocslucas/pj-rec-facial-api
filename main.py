from flask import Flask, request, jsonify
import logging
import os
import io
import numpy as np
from PIL import Image
import requests

# Ocultando logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#Importando DeepFace
from deepface import DeepFace

app = Flask(__name__)

# Função para verificar as imagens
def verify(img1_array, img2_array):
    try:
        logging.info("Iniciando a verificação facial...")
        resultado = DeepFace.verify(img1_array, img2_array)
        logging.info("Verificação concluída.")
        return {
            "verificado": resultado['verified'],
            "tempo_de_verificacao": resultado['time']
        }
    except Exception as e:
        logging.error(f"Erro na verificação facial, Por favor tire a foto novamente: {str(e)}")
        return {"erro": f"Erro na verificação facial: {str(e)}"}

# Função para baixar a imagem a partir de uma URL
def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logging.error(f"Erro ao baixar a imagem da URL {url}: {str(e)}")
        return None

# Rota para verificação
@app.route('/verificar', methods=['POST'])
def verificar_rosto():
    data = request.form  # Usando form para aceitar arquivos de imagem e campos de formulário

    if 'img1' not in request.files or 'url2' not in data:
        return jsonify({"erro": "A imagem e a URL da segunda imagem são necessárias!"}), 400

    try:
        # Obtendo a imagem capturada
        img1 = request.files['img1']
        img1 = Image.open(img1).convert("RGB")

        # Baixando a segunda imagem da URL pré-assinada
        img2_url = data['url2']
        img2 = download_image(img2_url)
        
        if img2 is None:
            return jsonify({"erro": "Erro ao baixar a segunda imagem"}), 500

        # Convertendo as imagens para numpy arrays
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        # Verificação facial
        resultado = verify(img1_array, img2_array)
        return jsonify(resultado)

    except Exception as e:
        logging.error(f"Erro ao processar imagens: {str(e)}")
        return jsonify({"erro": "Erro ao processar as imagens"}), 500

if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    app.run(host=host, port=port, debug=False)
