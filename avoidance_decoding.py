import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from collections import defaultdict
import json
sys.path.append("C:/Users/Nadiyah Ahmed/OneDrive/Desktop/ad/")
from negative_sampling import NegativeSamples
from model import AvoidanceDecodingModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, help = "Path to data file")
    parser.add_argument("--model", type = str, required = True, help = "Model name or path")
    parser.add_argument("--decoding_len", type = int, default = 50, help = "Model name or path")
    parser.add_argument("--output_negative_samples_file", type = str, help = "File to save negative samples in")
    parser.add_argument("--negative_samples_file", type = str, help = "Path to negative samples file")
    parser.add_argument("--output_dir", type = str, required = True, help = "Output directory to save results")
    parser.add_argument("--beta", type = int, default = 2, help = "Beta value for similarity penalty scaling")
    parser.add_argument("--delta", type = float, default = 0.5, help = "to maintain sufficient low level diversity throughout the generation process based on empirical tuning")
    parser.add_argument("--T0", type = int, default = 25)
    

    args = parser.parse_args()

    #so what is the next step from here?
    #to read the prompts from the dataset and collect negative samples essentially
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype = torch.bfloat16,
                                                 device_map = "auto",
                                                 trust_remote_code=True) 
                                                #  attn_implementation = "flash_attention_2")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # set pad token if it doesnt exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #get negative samples' hidden states and sentence embeddings
    negative_samples = NegativeSamples(model, tokenizer)
    if args.data:
      neg_hidden_states, neg_sent_embeddings = negative_samples.generate_negative_samples(c_prompts = args.data, out_dir = args.output_negative_samples_file)

    if args.negative_samples_file:
      neg_hidden_states, neg_sent_embeddings = negative_samples.get_hidden_states_and_sent_embeddings(args.negative_samples_file)

    print("Negative hidden states and sentence embeddings computed")

    ad_model = AvoidanceDecodingModel(model = model, tokenizer = tokenizer, pad_token = "<END>", beta = args.beta, delta = args.delta)
    print("Avoidance Model Initialised")
    sys_prompt = "Act as a master storyteller. Write a 500 word story based on the following prompt."

    responses = defaultdict(list)

    for i in range(3):
        prompts = negative_samples.prompts

        for prompt in prompts:
            template = [{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": prompt}]
          
            input_ids = tokenizer.apply_chat_template(template, 
                                                    add_generation_prompt = False, 
                                                    return_tensors = "pt", 
                                                    # temperature = 0.7
                                                    padding = False, 
                                                    truncation = True)
            input_ids = input_ids.to(model.device)
            output = ad_model.avoidance_decoding_search(input_ids, neg_hidden_states[prompt], neg_sent_embeddings[prompt], args.decoding_len)
            output = torch.tensor(output).unsqueeze(0)
            text = tokenizer.decode(output, skip_special_tokens = True)

            print(f"Prompt:\n{prompt} ", "The generated text is:\n", text)
            
            responses[prompt].append(text)
            negative_samples.samples[prompt].append(text)
            negative_samples.hidden_states[prompt].append(negative_samples.get_hidden_states(output))
            negative_samples.sent_embeddings[prompt].append(negative_samples.get_sentence_embeddings(text))
            with open(args.out_dir, "w") as f:
              json.dump(responses, f)
          



# okay negative samples have been collected now what?
# use these negative samples to test avoidance decoding

#any specific parts that need to be tested with avoidance decoding?
# or do we directly jump into testing the main algorithm

#lets directly jumo into testing the main algorithm because only then other issues will come forward I guess
# once testing done, then write code for calculating evaluation metrics as mentioned in the paper


