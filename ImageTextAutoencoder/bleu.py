from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import torch
import torch.nn as nn

import torch.optim as optim
from models.stack_gan2.model1 import encoder_resnet, text_models, G_NET

model_ft = text_models.AutoEncoderD(config, embeddings_matrix)
model_ft.cuda()
dec = G_NET()
dec.cuda()
enc = encoder_resnet()
enc.cuda()

print("################# PATHS #####################")
gOG = "/home/tuomas_pyorre/Unlabeled_Captioning/evaluation/saved_models/original/netGIT_22000.pth"
dOG = "/home/tuomas_pyorre/Unlabeled_Captioning/evaluation/saved_models/original/netDIT.pth"

gCLIP = "/home/tuomas_pyorre/Unlabeled_Captioning/evaluation/saved_models/with_clip/netGIT_12000.pth"
dCLIP= "/home/tuomas_pyorre/Unlabeled_Captioning/evaluation/saved_models/with_clip/netDIT.pth"

print("################# LOAD #####################")
modelG_C = torch.load(gOG)
modelG_CLIP_C = torch.load(gCLIP)

modelD_C = torch.load(dOG)
modelD_CLIP_C = torch.load(dCLIP)

modelG = TheModelClass()

modelG = modelG['state_dict']
print(modelG)
modelG.eval()
modelG_CLIP.eval()
modelD.eval()
modelD_CLIP.eval()