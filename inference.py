
from simpletransformers.classification import ClassificationModel

model = ClassificationModel(
                     'roberta', 
                     'best_model',
                     use_cuda=False)

predictions, raw_outputs = model.predict(['Obama wanted liberals &amp; illegals to move into red states'])
print(predictions)
        

