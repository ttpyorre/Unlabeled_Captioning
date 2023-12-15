from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import torch

print("################# PATHS #####################")
gOG = "/home/tuomas_pyorre/Unlabeled_Captioning/evaluation/models/original/netGIT_22000.pth"
dOG = "/home/tuomas_pyorre/Unlabeled_Captioning/evaluation/models/original/netDIT.pth"

gCLIP = "/home/tuomas_pyorre/Unlabeled_Captioning/evaluation/models/with_clip/netGIT_12000.pth"
dCLIP= "/home/tuomas_pyorre/Unlabeled_Captioning/evaluation/models/with_clip/netDIT.pth"

print("################# LOAD #####################")
modelG = torch.load(gOG)
modelG_CLIP = torch.load(gCLIP)

modelD = torch.load(dOG)
modelD_CLIP = torch.load(dCLIP)
modelG = modelG['state_dict']
print(modelG)
modelG.eval()
modelG_CLIP.eval()
modelD.eval()
modelD_CLIP.eval()