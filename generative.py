from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import math
import time
import sys
import json
import numpy as np

from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import evaluate

answers = ['A','B','C','D']

def main():
  # Preprocessing
  # Read a jsonl file and return each raw prompt (fact+question stem+A+B+C+D+correctAnswer)
  def get_prompts(fname):
    prompts = []
    with open(fname) as json_file:
      json_list = list(json_file)
      for i in range(len(json_list)):
          json_str = json_list[i]
          result = json.loads(json_str)
          
          base = result['fact1'] + ' ' + result['question']['stem']
          ans = answers.index(result['answerKey'])

          for j in range (4):
            base = base + ' [' + answers[j] + '] ' + result['question']['choices'][j]['text']

          base = tokenizer.eos_token + base + tokenizer.eos_token + ' ['+ answers[ans] + f'] ' + result['question']['choices'][ans]['text'] + tokenizer.eos_token
          #base = tokenizer.eos_token + base + ' ' + result['question']['choices'][ans]['text'] + tokenizer.eos_token
          prompts.append(base)

    return prompts

  class QADataset(Dataset):
      def __init__(self, prompts, tokenizer, max_length):
          self.input_ids = []
          #self.attention_mask = []
          
          for prompt in prompts:
              encodings_dict = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
              self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
              #self.attention_mask.append(torch.tensor(encodings_dict['attention_mask']))

      def __len__(self):
          return len(self.input_ids)

      def __getitem__(self, idx):
          return self.input_ids[idx]#, self.attention_mask[idx]
      

  def load_test_prompts():
    prompts = []
    all_answers = []
    correct_answers = []
    with open('test_complete.jsonl') as json_file:
      json_list = list(json_file)
      for i in range(len(json_list)):
          json_str = json_list[i]
          result = json.loads(json_str)
          
          base = result['fact1'] + ' ' + result['question']['stem']
          ans = answers.index(result['answerKey'])
          answer_set = []

          for j in range (4):
            base = base + ' [' + answers[j] + '] ' + result['question']['choices'][j]['text']
            answer_set.append(' [' + answers[j] + '] ' + result['question']['choices'][j]['text'])

          base = tokenizer.eos_token + base + tokenizer.eos_token 
          #asw = answers[ans] + '- ' + result['question']['choices'][ans]['text']
          #base = tokenizer.eos_token + base + ' ' 
          #asw = ' [' + answers[ans] + '] ' + result['question']['choices'][ans]['text']
          prompts.append(base)
          all_answers.append(answer_set)
          correct_answers.append(ans)

    return prompts, all_answers, correct_answers
  
  def load_validtest_prompts():
    prompts = []
    all_answers = []
    correct_answers = []
    with open('dev_complete.jsonl') as json_file:
      json_list = list(json_file)
      for i in range(len(json_list)):
          json_str = json_list[i]
          result = json.loads(json_str)
          
          base = result['fact1'] + ' ' + result['question']['stem']
          ans = answers.index(result['answerKey'])
          answer_set = []

          for j in range (4):
            base = base + ' [' + answers[j] + '] ' + result['question']['choices'][j]['text']
            answer_set.append(' [' + answers[j] + '] ' + result['question']['choices'][j]['text'])

          base = tokenizer.eos_token + base + tokenizer.eos_token 
          #asw = answers[ans] + '- ' + result['question']['choices'][ans]['text']
          #base = tokenizer.eos_token + base + ' ' 
          #asw = ' [' + answers[ans] + '] ' + result['question']['choices'][ans]['text']
          prompts.append(base)
          all_answers.append(answer_set)
          correct_answers.append(ans)

    return prompts, all_answers, correct_answers


  def get_answers_fine_tune(_prompts):
    predicted_answers = []

    for i in range(len(_prompts)):
      input_ids = tokenizer.encode(_prompts[i], return_tensors='pt')
      # get predicted answer
      beam_output = model2.generate(
        input_ids.to(device), 
        max_length = len(torch.squeeze(input_ids)) + 20,
        num_beams=5, 
        # early_stopping=True
      )

      # clean the answer
      # the output is always in the following format: eos + prompt + eos + [label of correct answer] + correct answer + [..] ... noises; 
      # I attempt to extract '[label of correct answer] + correct answer'
      # id50256: eos token, id685: ' ['

      # Try to find the position of the second eos

      output = beam_output[0]

      pos_eos = (beam_output[0]==50256).nonzero().squeeze()
      if len(pos_eos) > 1:
        pos1 = pos_eos[1]
        output = beam_output[0][pos1 + 1:]

      # Try to find the position of the second '['
      pos_brac = torch.squeeze((output==685).nonzero(),dim = 1)

      if len(pos_brac) > 1:
        pos2 = pos_brac[1]
        output = output[:pos2] 

      # test_predicted is the predicted answer. format: ' [label] + answer'
      test_predicted = tokenizer.decode(output)
      predicted_answers.append(test_predicted)

    return predicted_answers


  def get_answers_zero_shot(_prompts):

    predicted_answers = []

    #load a new gpt2 model
    model3 = GPT2LMHeadModel.from_pretrained('gpt2')
    model3 = model3.to(device)
    model3.resize_token_embeddings(len(tokenizer))


    for i in range(len(_prompts)):
      input_ids = tokenizer.encode(_prompts[i], return_tensors='pt')
      # get predicted answer
      beam_output = model3.generate(
        input_ids.to(device), 
        max_length = len(torch.squeeze(input_ids)) + 20,
        num_beams=5, 
        # early_stopping=True
      )

      output = beam_output[0]
      test_predicted = tokenizer.decode(output)
      predicted_answers.append(test_predicted)

    return predicted_answers

  # helper function to get arg_max of a list
  def arg_max(l):
    idx = 0
    for i in range(len(l)):
      if l[i] > l[idx]:
        idx = i
    return idx

  def evaluate_accuracy(predicted_answers,ans_type, _prompts, _all_answers, _labels):
    metric = evaluate.load("rouge")

    raw_counts = 0.0
    r1_counts = 0.0
    r2_counts = 0.0
    rL_counts = 0.0


    for i, predicted_answer in enumerate(predicted_answers):
      # the predicted answer is in the format ' [label] + answer' ; predicted_answer[2] is the label (A/B/C/D)
      if predicted_answer[2] == answers[_labels[i]]:
        raw_counts += 1

      # compare the predicted answer with each of the four choices
      # https://huggingface.co/spaces/evaluate-metric/rouge
      r1_scores = []
      r2_scores = []
      rL_scores = []

      for j in range(4):
        results = metric.compute(predictions = [predicted_answer],references = [_all_answers[i][j]])
        r1_scores.append(results['rouge1'])
        r2_scores.append(results['rouge1'])
        rL_scores.append(results['rougeL'])

      # make the prediction for this prompt
      r1_argmax = arg_max(r1_scores)
      r2_argmax = arg_max(r2_scores)
      rL_argmax = arg_max(rL_scores)

      if r1_argmax == _labels[i]:
        r1_counts += 1
      if r2_argmax == _labels[i]:
        r2_counts += 1
      if rL_argmax == _labels[i]:
        rL_counts += 1


    if ans_type == 'fine-tune':  
      print(ans_type+' accuracy(raw_label): ', raw_counts/len(_prompts))


    print(ans_type+' accuracy(r1): ', r1_counts/len(_prompts))
    print(ans_type+' accuracy(r2): ', r2_counts/len(_prompts))
    print(ans_type+' accuracy(rL): ', rL_counts/len(_prompts))
  
  # start
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token

  # prepare train and valid prompts (type: list)
  # Train and valid prompts are different from test prompts below in that they contain the four options and the correct answer
  train_prompts = get_prompts('train_complete.jsonl')
  valid_prompts = get_prompts('dev_complete.jsonl')

  
  tokenized_train = QADataset(train_prompts, tokenizer, 128)
  tokenized_valid = QADataset(valid_prompts, tokenizer, 128)


  # prepare model, data_collator, and training_args
  model2 = GPT2LMHeadModel.from_pretrained('gpt2')
  model2.resize_token_embeddings(len(tokenizer))
  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

  args = TrainingArguments(
      output_dir="generative-model",
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      evaluation_strategy="steps",
      eval_steps = 2_00,
      logging_steps=2_00,
      gradient_accumulation_steps=2,
      num_train_epochs=8,
      weight_decay=0.1,
      warmup_steps=5_00,
      lr_scheduler_type="cosine",
      learning_rate=5e-4,
      save_steps=5_00,
      fp16=True,
      push_to_hub=False,
  )

  # set up trainer
  trainer = Trainer(
      model=model2,
      tokenizer=tokenizer,
      args=args,
      data_collator=data_collator,
      train_dataset=tokenized_train,
      eval_dataset=tokenized_valid,
  )

  print(train_prompts[0])

  trainer.train()

  # prepare testing set for evaluation
  test_prompts, test_all_answers, test_labels = load_test_prompts()
  validtest_prompts, validtest_all_answers, validtest_labels = load_test_prompts()


  test_fine_tune_answers = get_answers_fine_tune(test_prompts)
  test_zero_shot_answers = get_answers_zero_shot(test_prompts)
  
  # print out the accuracy
  evaluate_accuracy(test_fine_tune_answers, 'fine-tune', test_prompts, test_all_answers, test_labels)
  evaluate_accuracy(test_zero_shot_answers, 'zero-shot', test_prompts, test_all_answers, test_labels)

  validtest_fine_tune_answers = get_answers_fine_tune(validtest_prompts)
  validtest_zero_shot_answers = get_answers_zero_shot(validtest_prompts)
  
  # print out the accuracy
  evaluate_accuracy(validtest_fine_tune_answers, 'fine-tune', validtest_prompts, validtest_all_answers, validtest_labels)
  evaluate_accuracy(validtest_zero_shot_answers, 'zero-shot', validtest_prompts, validtest_all_answers, validtest_labels)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device available for running: ")
  print(device)
  torch.cuda.empty_cache()

  main()