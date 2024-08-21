import os
import sys
import logging
import urllib.request
from laserembeddings import Laser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LASERMODEL:
    def __init__(self,output_dir):
        logging.info("Loading LASER models ")
        self.download_models("./ContextWordEmbedding/Model/Laser")
        DEFAULT_BPE_CODES_FILE = os.path.join(output_dir, '93langs.fcodes')
        DEFAULT_BPE_VOCAB_FILE = os.path.join(output_dir, '93langs.fvocab')
        DEFAULT_ENCODER_FILE = os.path.join(output_dir,
                                            'bilstm.93langs.2018-12-26.pt')
        self.embedding_model = Laser(DEFAULT_BPE_CODES_FILE, DEFAULT_BPE_VOCAB_FILE, DEFAULT_ENCODER_FILE)
        
    def embeddings(self,x):
        return self.embedding_model.embed_sentences(x,lang='en')
    
    def download_file(self,url, dest):
        sys.stdout.flush()
        urllib.request.urlretrieve(url, dest)

def download_models(self,output_dir):
    logger.info('Downloading models into {}'.format(output_dir))

    self.download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes',
                os.path.join(output_dir, '93langs.fcodes'))
    self.download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab',
                os.path.join(output_dir, '93langs.fvocab'))
    self.download_file(
        'https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt',
        os.path.join(output_dir, 'bilstm.93langs.2018-12-26.pt'))
    
    
if __name__ == "__main__":
    output_dir = "./Models/Laser"
    model = LASERMODEL(output_dir)
    download_models(output_dir)
    sentences = ["Language modeling is gifted to the community "]
    anchor_word= "gifted"
    embeddings = model.embeddings(sentences)
    print(embeddings.shape)
    