from collections import defaultdict
from pickle import NONE
import torch
import os
import json
from utils import get_sentence_embeddings

'''Let this class save the hidden states and embeddings for the negative samples so we don't have to compute them everytime during decoding'''
class NegativeSamples:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = []
        self.samples = defaultdict(list)
        self.hidden_states = defaultdict(list)
        self.sent_embeddings = defaultdict(list)
        self.output_dir = NONE

    def read_prompts(self, data_file):
        with open(data_file, "r") as f:
            prompts = f.readlines()
        
        for prompt in prompts:
            self.prompts.append(prompt.strip())

        self.prompts = [p for p in self.prompts if p!=""]


    def generate_negative_samples(self, c_prompts, out_dir, num_samples = 10):
      self.output_dir = out_dir
      os.makedirs(self.output_dir, exist_ok = True)

      if ".txt" in c_prompts:
          self.read_prompts(c_prompts)

      else:
          for prompt in c_prompts:
              self.prompts.append(prompt)
      
      print(self.prompts)
      
      for prompt in self.prompts:
        sys_prompt = f'''Act as a master storyteller. Write a 500 word story based on the prompt below. '''   
      
        if prompt not in self.samples:   

            for i in range(num_samples):
              # sys_prompt = sys_prompt + f"The story should not have the same characters, plot or setting as the previously generated stories. The previously generated stories are {self.samples[prompt]}"      
              template = [{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": f"The prompt is {prompt}"}] 
              input_ids = self.tokenizer.apply_chat_template(template,
                                                          add_generation_prompt = False,
                                                          return_tensors = "pt",
                                                          padding = False,
                                                          truncation = True)

                  # input_ids = self.tokenizer.encode(final_prompt, 
                  #                                   padding = False,
                  #                                   truncation = True,
                  #                                   max_length = 700,
                  #                                   return_tensors = "pt")
                  
              input_ids = input_ids.to(self.model.device)
              with torch.no_grad():
                  outputs = self.model.generate(**input_ids,
                                              max_new_tokens = 512,
                                              temperature = 0.9,
                                              pad_token_id = self.tokenizer.pad_token_id,
                                              eos_token_id = self.tokenizer.eos_token_id,
                                              use_cache = True)
                  
              # compute hidden states and add
              self.hidden_states[prompt].append(self.get_hidden_states(input_ids))
                  
              # decode the generated tokens

              response = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
              story_response = self.clean_samples(response)
              # get sentence embeddings and add
              self.sent_embeddings(self.get_sentence_embedding(story_response))
              self.samples[prompt].append(story_response)

      with open(f"{self.output_dir}/negative_samples.json", "w") as f:
        json.dump(self.samples, f)

      return self.samples

    def clean_samples(self, res, split_point = "</think>\n\n</think>"):
        res = res.strip("\n").split(split_point)
        return res[-1]

    def get_hidden_states(self, input_ids = None):
      if input_ids:
        output = self.model(input_ids, output_hidden_states = True)
        last_hidden_states = output.hidden_states[-1]
        return last_hidden_states

      else:
        for prompt in self.prompts:
          for sample in self.samples[prompt]:
            input_ids = self.tokenizer.encode(sample, return_tensors = "pt")
            input_ids = input_ids.to(self.model.device)
            with torch.no_grad():
              output = self.model(input_ids, output_hidden_states = True)
            self.hidden_states[prompt].append(output.hidden_states[-1])

            

    def get_sentence_embeddings(self, text = None):
      #load a sentence transformer which will be used to convert the text to embeddings
      if text:
        return get_sentence_embeddings(text)
  
      else: 
        for prompt in self.prompts:
          for sample in self.samples[prompt]:
            sent_embed = get_sentence_embeddings(sample)
            self.sent_embeddings[prompt].append(sent_embed)



    def get_hidden_states_and_sent_embeddings(self, samples_file):
      if "json" in samples_file:
        with open(samples_file, "r") as f:
          negative_samples = json.load(f)

      self.prompts = negative_samples.keys()
      for prompt in self.prompts:
        self.samples[prompt] = negative_samples[prompt]

      self.get_hidden_states()
      self.get_sentence_embeddings()

      return self.hidden_states, self.sent_embeddings


    