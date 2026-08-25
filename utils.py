import torch
from torch.nn import functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
sent_transformer = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


def enlarge_past_key_values(past_key_values, k):
        # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
        new_key_values = []
        for layer in past_key_values:
            items = []
            for item in layer:
                # item is the key and value matrix
                bsz, num_head, seq_len, esz = item.size()
                item = item.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(bsz*k, num_head, seq_len, esz)
                item = item.float().cpu().detach().numpy()
                items.append(item)
            items = torch.from_numpy(np.array(items))    
            # new_key_values.append(torch.tensor(items))
            new_key_values.append(items)
        
        new_key_values = torch.from_numpy(np.array(new_key_values)) 
        return new_key_values
        


def get_next_top_k(model, input_ids, k):
    # bsz, seq_len, num_head, esz = input_ids.size()

    outputs = model(input_ids, output_hidden_states = True, use_cache = True)

    logits_for_next_step = outputs.logits[:, -1, :]

    past_key_values = outputs.past_key_values

    _, top_k_ids = torch.topk(logits_for_next_step, dim = -1, k = k)
    probs = F.softmax(logits_for_next_step) 
    top_k_probs = torch.gather(probs, dim = -1, index = top_k_ids)


    # past_key_values = enlarge_past_key_values(past_key_values, k)
    # past_key_values = past_key_values.unsqueeze(1).reshape(-1, k, -1, -1)
    # item = item.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(bsz*k, num_head, seq_len, esz)

    outputs_for_top_k = model(top_k_ids,
                         past_key_values = past_key_values,
                         output_hidden_states = True,
                         use_cache = True)


    return top_k_ids, top_k_probs, outputs_for_top_k
    

def hybrid_similarity_penalty(model, tokenizer, input_ids, neg_hidden_states, neg_sent_embeddings, k, alpha, gamma, beta = 2.0):
    top_k_ids, top_k_probs, outputs_for_top_k = get_next_top_k(model, input_ids, k)
    bsz, len_ = input_ids.size()
    # need to see if we can loop over the sentence embeddings and hidden_states of the cand keys 
    candidate_sentence_embeddings = []
    # for id_ in top_k_ids:
    # sentence = tokenizer.decode(top_k_ids, add_special_tokens = False)
    # candidate_sentence_embeddings.append(get_sentence_embeddings(sentence))
    # print(input_ids)
    tokens = top_k_ids.view(k, 1)
    expanded_context = input_ids.expand(k, -1)
    full_sequences = torch.cat([expanded_context, tokens], dim=-1)
    full_sents = tokenizer.decode(full_sequences, skip_special_tokens = True)
    candidate_sentence_embeddings = get_sentence_embeddings(full_sents)

    # for id_ in top_k_ids[0]:
    #   context = input_ids[0]+[id_]
    #   print(context)
    #   context = context.unsqueeze(0).to(model.device)
    #   decoded = tokenizer.decode(context, skip_special_tokens = True)
    #   print(decoded)
    #   candidate_sentence_embeddings.append(get_sentence_embeddings(decoded).float().detach().cpu())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_hidden_states = outputs_for_top_k.hidden_states[-1]


    N = len(neg_sent_embeddings)

    # for every token, k score with every negative sample, N
    '''Putting it here to remember it -
    CSP as maximum cosine similarity between last hidden representations of candidate tokens and those of all individual tokens of negative samples.'''
    # normalize hidden states
    norm_candidate_hidden = candidate_hidden_states / candidate_hidden_states.norm(dim = 2, keepdim = True)
    # norm_candidate_hidden = norm_candidate_hidden.squeeze()
    norm_negative_hidden = []
    for hidden_state in neg_hidden_states:
      norm_hidden_state = hidden_state / hidden_state.norm(dim = 2, keepdim = True)
      norm_negative_hidden.append(norm_hidden_state)

    top_k_probs = top_k_probs.float().detach().cpu()  
    # top_k_probs = top_k_probs.squeeze(0)

    csp_scores = torch.zeros((k, N)) 
    nsp_scores = torch.zeros((k, N))
    hybrid_penalty_scores = torch.zeros((k, N))
    final_similarity_scores = torch.zeros(k) 
    final_scores= torch.zeros(k)
    for j in range(k):
        csp_scores = torch.zeros(N)
        nsp_scores = torch.zeros(N)
        hybrid_penalty_scores = torch.ones(N)
        for i in range(N):
          csp = [F.cosine_similarity(norm_candidate_hidden[:, j, :], t.unsqueeze(0)).float().detach().cpu() for t in norm_negative_hidden[i][0]]
          csp_scores[i] = torch.max(torch.from_numpy(np.array(csp)))
          nsp_scores[i] = F.cosine_similarity(candidate_sentence_embeddings[j, :], neg_sent_embeddings[i], dim = 0)
          hybrid_penalty_scores[i] = (gamma*csp_scores[i]) + ((1-gamma)*nsp_scores[i])

        final_similarity_scores[j] = torch.max(beta*hybrid_penalty_scores)
        final_scores[j] = (1-alpha)*top_k_probs[:, j] - (alpha*final_similarity_scores[j])

    return final_scores



def ranking(model, tokenizer, input_ids, neg_hidden_states, neg_sent_embeddings, k, alpha, gamma, beta):
     
    final_scores = hybrid_similarity_penalty(model, tokenizer, input_ids, neg_hidden_states, neg_sent_embeddings, k, alpha, gamma, beta)
    index = torch.argmax(final_scores)
    return index



def avoidance_decoding(model, tokenizer, input_ids, neg_hidden_states, neg_sent_embeddings, k, alpha, gamma, beta, decoding_len):
    
    outputs = model(input_ids, output_hidden_states = True, use_cache = True)

    logits = outputs.logits
    last_hidden_states = outputs.hidden_states[-1]
    bsz, seqlen, hidden_size = last_hidden_states.size()
    
    top_k_ids, top_k_probs, outputs_for_top_k = get_next_top_k(model, input_ids, k)
    # print(outputs_for_top_k.logits.size())

    past_key_values = outputs_for_top_k.past_key_values
    # logits = outputs_for_top_k.logits[:, -1, :]    # [B*K, V]
    logits = outputs_for_top_k.logits
    next_hidden = outputs_for_top_k.hidden_states[-1]    # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, k, -1, -1).reshape(bsz*k, seqlen, hidden_size)    # [B*K, S, E]


    selected_index = ranking(model, tokenizer, input_ids, neg_hidden_states, neg_sent_embeddings, k, alpha, gamma, beta)
    print(f"The selected index is: {selected_index}")
    next_id = top_k_ids[range(len(top_k_ids)), selected_index].unsqueeze(-1)    # [B, 1]
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), k))    # [B, K, E]
    next_hidden = next_hidden.squeeze(0)
    next_hidden = next_hidden[range(bsz), selected_index, :]    # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
    past_key_values = select_past_key_values(past_key_values, k, selected_index)
    logits = torch.stack(torch.split(logits, k)).squeeze(0)
    logits = logits[range(bsz), selected_index, :]    # [B, V]
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states, logits 
    
    # in order to calculate the hidden states for the candidate token we first need to get the candidate tokens and then use them to get logits



def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
          if item is not None:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz]
            item = item.squeeze(0)
            item = item[range(bsz), :, selected_idx, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values




def get_sentence_embeddings(text):
  with torch.no_grad():
    embedding = sent_transformer.encode(text, 
                                      batch_size = 1,
                                      precision = "float32",
                                      convert_to_numpy = False,
                                      convert_to_tensor="pt",
                                      normalize_embeddings = True)
  return embedding