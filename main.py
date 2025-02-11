from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import logging
from werkzeug.utils import secure_filename

# Configuração para esconder logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

# Função para verificar as imagens
def verify(img1_path, img2_path):
    try:
        logging.info(f"Iniciando a verificação facial entre {img1_path} e {img2_path}...")
        resultado = DeepFace.verify(img1_path, img2_path)
        logging.info("Verificação concluída.")
        return {
            "verificado": resultado['verified'],
            "tempo_de_verificacao": resultado['time']
        }
    except Exception as e:
        logging.error(f"Erro na verificação facial: {str(e)}")
        return {"erro": f"Erro na verificação facial: {str(e)}"}

# Função auxiliar para salvar os arquivos recebidos
def salvar_imagem(imagem, nome_arquivo):
    caminho = os.path.join('rostos_img', nome_arquivo)
    imagem.save(caminho)
    return caminho

# Rota para verificação de rostos
@app.route('/verificar', methods=['POST'])
def verificar_rosto():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"erro": "As duas imagens são necessárias!"}), 400
    
    try:
        # Lendo as imagens diretamente da requisição
        img1 = request.files['img1']
        img2 = request.files['img2']
        
        # Salvar as imagens no diretório "rostos_img"
        img1_path = salvar_imagem(img1, secure_filename(img1.filename))
        img2_path = salvar_imagem(img2, secure_filename(img2.filename))
        
        # Verificar se as imagens existem
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            resultado = verify(img1_path, img2_path)
            return jsonify(resultado)
        else:
            return jsonify({"erro": "Erro ao salvar as imagens!"}), 500
    except Exception as e:
        logging.error(f"Erro ao processar imagens: {str(e)}")
        return jsonify({"erro": "Erro ao processar as imagens"}), 500

# Inicializando a API
if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    app.run(host=host, port=port, debug=False)
