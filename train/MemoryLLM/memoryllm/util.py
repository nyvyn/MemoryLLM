import importlib

import torch
import numpy as np



def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



# def collate_fn(data, tokenizer, max_length, num_tokens, 
#                     add_special_tokens=False, 
#                     end_special_token=None, 
#                     mask_strategy='word', 
#                     mask_ratio=0.0,
#                     padding='longest'):

#     data = list(zip(*data))

#     if len(data) == 4:
#         contexts, sentences, target_is_context_indicator, labels = data
#         unrelated_contexts = None

#     else:
#         unrelated_contexts, contexts, sentences, target_is_context_indicator, labels = data

#     target_is_context_indicator = torch.tensor(target_is_context_indicator)

#     if end_special_token is not None:
#         sentences = [x + end_special_token for x in list(sentences)]
    
#     contexts_tokenized = tokenizer(list(contexts), 
#                                    max_length=max_length, 
#                                    padding=padding,
#                                    truncation=True, 
#                                    return_tensors='pt',
#                                    add_special_tokens=add_special_tokens)

#     if unrelated_contexts is not None:
#         unrelated_contexts_tokenized = tokenizer(list(unrelated_contexts), 
#                                    max_length=max_length, 
#                                    padding=padding,
#                                    truncation=True, 
#                                    return_tensors='pt',
#                                    add_special_tokens=add_special_tokens)

#     sentences_tokenized = tokenizer(list(sentences), 
#                                     max_length=max_length, 
#                                     truncation=True, 
#                                     padding='longest',
#                                     return_tensors='pt',
#                                     add_special_tokens=add_special_tokens)

#     sentences_ids = sentences_tokenized.input_ids
#     sentences_attention_mask = sentences_tokenized.attention_mask

#     context_ids = contexts_tokenized.input_ids
#     contexts_attention_mask = contexts_tokenized.attention_mask
    
#     if unrelated_contexts is not None:
#         unrelated_contexts_ids = unrelated_contexts_tokenized.input_ids
#         unrelated_contexts_attention_mask = unrelated_contexts_tokenized.attention_mask

#     # mask contexts and unrelated_contexts
#     if mask_ratio > 0.0:
#         context_ids[torch.where(target_is_context_indicator==True)], contexts_attention_mask[torch.where(target_is_context_indicator==True)] = mask_tokens(context_ids[torch.where(target_is_context_indicator==True)], contexts_attention_mask[torch.where(target_is_context_indicator==True)], mask_strategy, mask_ratio, tokenizer)
#         if unrelated_contexts is not None:
#             unrelated_contexts_ids[torch.where(target_is_context_indicator==True)], unrelated_contexts_attention_mask[torch.where(target_is_context_indicator==True)] = mask_tokens(unrelated_contexts_ids[torch.where(target_is_context_indicator==True)], unrelated_contexts_attention_mask[torch.where(target_is_context_indicator==True)], mask_strategy, mask_ratio, tokenizer)

#     # Create attention masks with total_length
#     contexts_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1), 
#                                          contexts_attention_mask], dim=-1)
#     if unrelated_contexts is not None:
#         unrelated_contexts_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1),
#                                                     unrelated_contexts_attention_mask], dim=-1)
#     sentences_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1),
#                                           sentences_attention_mask], dim=-1)
#     if unrelated_contexts is not None:
#         return context_ids, contexts_attention_mask, sentences_ids, sentences_attention_mask, unrelated_contexts_ids, unrelated_contexts_attention_mask, torch.tensor(labels)
#     else:
#         return context_ids, contexts_attention_mask, sentences_ids, sentences_attention_mask, torch.tensor(labels)

# # Then when you create the DataLoader:
# collate_fn_with_params = partial(collate_fn, tokenizer=tokenizer, max_length=max_length, total_length=total_length)
# data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_with_params)

def add_context_to_list(new_context_ids, cur_context_ids, all_contexts, max_length):

    # new_context_ids: new_sentence
    # cur_context_ids: the sentence concatenated so far

    if len(new_context_ids) + len(cur_context_ids) > max_length:

        if len(cur_context_ids) > 0:
            all_contexts.append(cur_context_ids)

        while len(new_context_ids) > max_length:
            all_contexts.append(new_context_ids[:max_length])
            new_context_ids = new_context_ids[max_length:]
        
        cur_context_ids = new_context_ids
    
    else:
        cur_context_ids = torch.cat([cur_context_ids, new_context_ids])


    return cur_context_ids.long()

def collate_fn_qa(data, tokenizer, max_length, num_tokens, 
                eval_max_length=None,
                add_special_tokens=False, 
                end_special_token=None,
                mask_strategy=None,
                mask_ratio=None,
                padding='longest'):
                
    eval_max_length = max_length if eval_max_length is None else eval_max_length

    contexts, questions, answers, unrelated_contexts = zip(*data)

    if end_special_token is not None:
        answers = [x + end_special_token for x in list(answers)]
    
    contexts_tokenized = tokenizer(list(contexts), 
                                   max_length=max_length, 
                                   padding=padding,
                                   truncation=True, 
                                   return_tensors='pt',
                                   add_special_tokens=add_special_tokens)
    questions_tokenized = tokenizer(list(questions), 
                                    max_length=eval_max_length,
                                    truncation=True, 
                                    padding='longest',
                                    return_tensors='pt',
                                    add_special_tokens=add_special_tokens)
    answers_tokenized = tokenizer(list(answers),
                                    max_length=eval_max_length,
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt',
                                    add_special_tokens=add_special_tokens)

    # eg: batch_size: 4
    # eg: time_steps: 8
    # then unrelated_contexts would be 4 * 8; 
    # I need it to be 8 * 4

    unrelated_contexts = np.array(unrelated_contexts).transpose().tolist()

    all_unrelated_contexts = {}
    all_unrelated_contexts_mask = {}
    for i in range(len(unrelated_contexts)):
        all_unrelated_contexts[i] = tokenizer(unrelated_contexts[i],
                                    max_length=max_length,
                                    # padding='max_length',
                                    truncation=True,
                                    padding=padding,
                                    return_tensors='pt',
                                    add_special_tokens=add_special_tokens)
        all_unrelated_contexts_mask[i] = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(all_unrelated_contexts[i].input_ids.shape[0], 1),
                            all_unrelated_contexts[i].attention_mask], dim=-1)

    # Create attention masks with total_length
    contexts_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1), 
                                         contexts_tokenized.attention_mask], dim=-1)
    questions_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1), 
                                          questions_tokenized.attention_mask], dim=-1)
    answers_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1),
                                         answers_tokenized.attention_mask], dim=-1)

    outputs = (contexts_tokenized.input_ids, contexts_attention_mask, \
        questions_tokenized.input_ids, questions_attention_mask, \
        answers_tokenized.input_ids, answers_attention_mask)
    
    for i in range(len(all_unrelated_contexts)):
        outputs += (all_unrelated_contexts[i].input_ids,)
        outputs += (all_unrelated_contexts_mask[i],)
        
    return outputs



        answers_tokenized['input_ids'], answers_tokenized['attention_mask']

def calculate_exact_hit_accuracy(preds, targets):
    hit = 0
    for pred, target in zip(preds, targets):
        if target.replace("<s>", "") in pred:
            hit += 1
    return hit / len(preds)