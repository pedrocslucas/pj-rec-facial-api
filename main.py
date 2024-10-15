#########################################################################################
from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import logging

# Configuração para esconder logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

# Função de verificação de rostos
def verify(img1_path, img2_path):
    try:
        resultado = DeepFace.verify(img1_path, img2_path)
        verificacao = resultado['verified']
        time_taken = resultado['time']

        return {
            "verificado": verificacao,
            "tempo_de_verificacao": time_taken
        }
    except Exception as e:
        return {"erro": str(e)}

# Rota para verificação de rostos
@app.route('/verificar', methods=['POST'])
def verificar_rosto():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"erro": "As duas imagens são necessárias!"}), 400
    
    img1 = request.files['img1']
    img2 = request.files['img2']

    img1_path = os.path.join("temp", "img1.jpg")
    img2_path = os.path.join("temp", "img2.jpg")

    # Salva as imagens enviadas no diretório temporário
    img1.save(img1_path)
    img2.save(img2_path)

    # Verifica as imagens
    resultado = verify(img1_path, img2_path)

    # Remove as imagens temporárias após a verificação
    os.remove(img1_path)
    os.remove(img2_path)

    return jsonify(resultado)

# Inicializa a API
if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(host='0.0.0.0', port=5000)
