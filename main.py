from flask import Flask, request, jsonify
import logging
import os
import io
import numpy as np
from PIL import Image

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

        # Imgs como numpy arrays
        resultado = DeepFace.verify(img1_array, img2_array)

        logging.info("Verificação concluída.")
        return {
            "verificado": resultado['verified'],
            "tempo_de_verificacao": resultado['time']
        }
    except Exception as e:
        logging.error(f"Erro na verificação facial: {str(e)}")
        return {"erro": f"Erro na verificação facial: {str(e)}"}

# Rota para verificação
@app.route('/verificar', methods=['POST'])
def verificar_rosto():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"erro": "As duas imagens são necessárias!"}), 400

    try:
        # Lendo as imagens e convertendo para numpy arrays
        logging.info("Lendo img1...")
        img1 = Image.open(io.BytesIO(request.files['img1'].read())).convert("RGB")
        img1_array = np.array(img1)

        logging.info("Lendo img2...")
        img2 = Image.open(io.BytesIO(request.files['img2'].read())).convert("RGB")
        img2_array = np.array(img2)

        # Verificação facial
        resultado = verify(img1_array, img2_array)

        return jsonify(resultado)

    except Exception as e:
        logging.error(f"Erro ao processar imagens: {str(e)}")
        return jsonify({"erro": "Erro ao processar as imagens"}), 500

# Inicializando a API
if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    app.run(host=host, port=port, debug=False)
