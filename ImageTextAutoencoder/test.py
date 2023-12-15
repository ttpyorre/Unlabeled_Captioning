import pickle 

with open('birds_no_clip','rb') as file:
    flower_no_clip = pickle.load(file)

with open('birds_with_clip','rb') as file:
    flower_with_clip = pickle.load(file)

with open('BLEU birds_with_clip','rb') as file:
    bleu_flower_with_clip = pickle.load(file)

with open('BLEU birds_no_clip','rb') as file:
    bleu_flower_no_clip = pickle.load(file)
    
print('noclip', sum(flower_no_clip)/len(flower_no_clip))
print('withclip', sum(flower_with_clip)/len(flower_with_clip))
print('bleunoclip', sum(bleu_flower_no_clip)/len(bleu_flower_no_clip))
print('bleuwithclip', sum(bleu_flower_with_clip)/len(bleu_flower_with_clip))