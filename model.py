import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from scipy.stats import entropy
import numpy as np
from utils import hybrid_similarity_penalty, avoidance_decoding
''' This file will define the functions to calculate k and \alpha as calculated in the Adaptive Contrastive Similarity Paper
k: the number of tokens to be considered for contrastive search
alpha: the hyperparameter that decides the weight of model uncertainty and model degeneration

why do these parameters need to be calculated for avoidance decoding? 
refer to algorithm 1 in the paper'''

train_fct = CrossEntropyLoss()
class AvoidanceDecodingModel(nn.Module):
    def __init__(self, model, tokenizer, pad_token, beta, delta=0.5, gamma=None):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = pad_token
        # print("Add Pad Token to the tokenizer")
        self.tokenizer.add_tokens([pad_token])
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([pad_token])[0]
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.vocab_size = len(self.tokenizer)
        self.embed_dim = self.model.config.hidden_size
        self.alpha = None
        self.gamma = None
        self.beta = beta
        self.delta = delta
    
    def update_k_alpha(self, kld_uniform, kld_certain, smoothing = 1):
        delta = kld_uniform - kld_certain
        alpha = np.exp(delta/smoothing)/ (1+np.exp(delta/smoothing))
        k = int(np.round((10*alpha + 5)))
        return k, alpha

    
    
    def adaptive_k_and_alpha(self, logits):
        certainty_prob = 0.95
        uniform_dist = [1/len(self.tokenizer)]*len(self.tokenizer)
        # uniform_dist = [1/len(self.tokenizer) for i in range(0, len(self.tokenizer))]
        index = torch.argmax(logits[0, -1, :]).tolist() # word with the highest probability
        cert_prob = (1-certainty_prob)/len(self.tokenizer)
        cert_dist = [cert_prob for i in range(0, len(self.tokenizer))]
        cert_dist[index] += certainty_prob

        # the logit probabilites are taken as the truth distribution here to compare how much information model's logits contains compared to 
        # random distribution
        probs = F.softmax(logits[:, -1, :]).squeeze().float().detach().cpu().numpy()
        kld_uniform = entropy(probs, uniform_dist)

        # the logit probabilites are taken as the predicted values here to compare how well the model logits
        # match a certainty goal
        kld_cert = entropy(cert_dist, probs)

        self.k, self.alpha = self.update_k_alpha(kld_uniform, kld_cert)
        return self.k, self.alpha
    
    # for small t = wordcount
    def compute_gamma(self, t, T0 = 25):
        return self.delta + (1-self.delta)*(1/(1+np.exp(-(t-T0))))

    # According to the algorithm given in the paper, the next step is to calculate
    # similarity penalties, conceptual level similarity penalty and narrative level similarity penalty

    '''Conceptual Level Similarity Penalty = maximum cosine similarity between last hidden
    representations of candidate tokens and those of all individual tokens of negative samples'''

# for narrative level similarity penalty we calculate the similarity between new sentence embeddings and negative sentence embeddings
    
    def forward(self, input_ids, past_key_values, beta, neg_hidden_states, neg_sent_embeddings, k):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids = input_ids,
                             output_hidden_states = True,
                             past_key_values = past_key_values,
                             use_cache = True)
        logits = outputs.logits

        # # identify top_k
        # logits_for_next_step = logits[:, -1, :]
        # past_key_values = outputs.past_key_values
        # _, top_k_ids = torch.topk(logits_for_next_step, dim = -1, k = k)
        

        k, self.alpha = self.adaptive_k_and_alpha(logits)
        self.gamma = self.compute_gamma(input_ids)

        final_scores = hybrid_similarity_penalty(self.model, input_ids, neg_hidden_states, neg_sent_embeddings, k, self.alpha, self.gamma, beta)
        print("Final scores calculated")
        # the highest cosine similarity is 1 that means exactly like the negative samples, we want the similarities to be as low as possible right so 
        # what margin to take that values greater than that are penalised, lets take margin = 0.5 so if the similarity is more than this it would be penalised

        margin = 0.5

        penalty = F.relu(final_scores - margin)
        new_logits = logits - beta*penalty
        return new_logits

        
        # we would not need cross enropy loss here as we dont have labels so what remains is to see how to apply this decoding process now

    def avoidance_decoding_search(self, input_ids, neg_hidden_states, neg_sent_embeddings, decoding_len = 512):
        # bsz, seq_len, em_sz = input_ids.size()
        # generated = [item for item in input_ids.tolist()]
        generated = input_ids["input_ids"].tolist()[0]
        assert self.model.device == input_ids["input_ids"].device
        outputs = self.model(**input_ids, output_hidden_states = True, use_cache = True)
        prompt_ids = len(generated)
        print(prompt_ids)
        t = 0
        

        # if not self.gamma:
        #   self.gamma = self.compute_gamma(input_ids)
        k, self.alpha = self.adaptive_k_and_alpha(outputs.logits)

        for step in range(prompt_ids+decoding_len):
            
            new_input_ids = torch.tensor(generated).unsqueeze(0).to(self.model.device)
            # outputs = self.model(input_ids = new_input_ids, output_hidden_states = True)
            t +=1
            print(t)
            self.gamma = self.compute_gamma(t)

            print(f"k: {k}, alpha: {self.alpha}, gamma: {self.gamma}")

            input_ids, past_key_values, last_hidden_states, logits = avoidance_decoding(model = self.model,
                                                                                        tokenizer = self.tokenizer,
                                                                                        input_ids = new_input_ids,
                                                                                        neg_hidden_states = neg_hidden_states,
                                                                                        neg_sent_embeddings = neg_sent_embeddings,
                                                                                        k = k,
                                                                                        alpha = self.alpha,
                                                                                        gamma = self.gamma,
                                                                                        beta = self.beta,
                                                                                        decoding_len = decoding_len)
            
            token = input_ids.squeeze(dim = -1).tolist()
            print(token)
            # for idx, t in enumerate(tokens):
            #     generated[idx].append(t)
            #     print("Token: ", t)
            generated.extend(token)

            generated_text = self.tokenizer.decode(torch.tensor(generated), skip_special_tokens = True)
            print(generated_text)
            if token == self.tokenizer.eos_token_id:
              break

        return generated



        


    





    


    
    

