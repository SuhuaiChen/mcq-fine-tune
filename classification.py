from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import math
import time
import sys
import json
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report

class BERTfine(nn.Module):
    def __init__(self, zero_shot=False):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        if zero_shot:
          for param in self.model.parameters():
              param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,src):
        batch = self.tokenizer(src, return_tensors='pt', padding='longest', max_length=256)
        batch.to(device)
        batch_outputs = self.model(**batch)
        output = self.linear(batch_outputs.last_hidden_state[:,0,:])
        output = self.sigmoid(output)
        return output

def readJSON(filename, test_):
  res = []
  labels = []
  answers = ['A','B','C','D']
  with open(filename) as json_file:
      json_list = list(json_file)
      test = []
      for i in range(len(json_list)):
          json_str = json_list[i]
          result = json.loads(json_str)
          # concat fact1 with question
          base = result['fact1'] + ' [SEP] ' + result['question']['stem']
          ans = answers.index(result['answerKey'])
          test.append(ans)
          
          for j in range(4):
              text = base + ' ' + result['question']['choices'][j]['text']
              # text = tokenizer(text, return_tensors='pt')
              if j == ans:
                  label = 1
              else:
                  label = 0
              res.append(text)
              labels.append(label)
            
      if test_: labels = test
      return res, labels

def runBert(zero_shot):
  mode = "zero-shot" if zero_shot else "fine-tuned"
  ### Hyper parameters
  model = BERTfine(zero_shot=zero_shot)
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  criterion = nn.BCELoss()
  batch_size = 100
  num_workers = 2
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)

  ### create data loaders
  file_name = 'train_complete.jsonl'        
  train, train_labels = readJSON(file_name, False)
              
  file_name = 'dev_complete.jsonl'        
  valid, valid_labels = readJSON(file_name, False)

  valid_test, valid_test_labels = readJSON(file_name, True)
      
  file_name = 'test_complete.jsonl'        
  test, test_labels = readJSON(file_name, True)

  train_dataset = [(train[i], train_labels[i]) for i in range(len(train))]
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

  training_loss = []
  validation_loss = []

  ### main loop
  numEpochs = 4
  for epochs in range(numEpochs):
    print("training for epoch:", end="")
    print(epochs)
    train_loss = 0
    valid_loss = 0
    model.train()
    ## train loop
    for i, (batch, batch_labels) in enumerate(tqdm(train_loader)):
      if (i+1) % 1000 == 0:
          print("Batch completed: " + str(i+1))
      optimizer.zero_grad()
      output = model(batch)
      batch_labels = batch_labels.to(device).to(torch.float).unsqueeze(1)
      loss = criterion(output, batch_labels)
      loss.backward()
      train_loss += loss.item()
      optimizer.step()
    ## validation loop
    with torch.no_grad():
      model.eval()
      output = model(valid)
      _valid_labels = torch.Tensor(valid_labels).to(device).to(torch.float).unsqueeze(1)
      valid_loss = criterion(output, _valid_labels).item()
    scheduler.step()
    training_loss.append(train_loss/len(train_loader))
    validation_loss.append(valid_loss)
    print(train_loss/len(train_loader))
    print(valid_loss)

  torch.save(model.state_dict(), "classification_" + mode + ".pt")
#   loaded_model = MyModel()
#   loaded_model.load_state_dict(torch.load('model.pth'))

  ### run model on test set
  test_loader = DataLoader(test, batch_size=4, shuffle=False, num_workers=2)
  predict = []
  correct = 0

  for i, batch in enumerate(tqdm(test_loader)):
    output = model(batch)
    predict.append(torch.argmax(output).tolist())
    if test_labels[i] == predict[i]: correct += 1
  
  ### print results
  print(mode)
  print("Classification report for BERT model on test set:")
  print(classification_report(test_labels,predict))
  print(correct/len(test_labels))

  valid_test_loader = DataLoader(valid_test, batch_size=4, shuffle=False, num_workers=2)
  val_test_predict = []
  correct = 0

  for i, batch in enumerate(tqdm(valid_test_loader)):
    output = model(batch)
    val_test_predict.append(torch.argmax(output).tolist())
    if valid_test_labels[i] == val_test_predict[i]: correct += 1

  mode = "zero-shot" if zero_shot else "fine-tuned"
  print(mode)
  print("Classification report for BERT model on valid set:")
  print(classification_report(valid_test_labels,val_test_predict))
  print(correct/len(valid_test_labels))

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device available for running: ")
  print(device)
  
  runBert(zero_shot=False)
  runBert(zero_shot=True)