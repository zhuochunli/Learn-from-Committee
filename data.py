import json
import torch
import os
import numpy as np
import random
from datasets import load_dataset, DatasetDict
from all_prompts import *
import argparse
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from collections import defaultdict

# Uncomment the following to debug data.py individually, without train.py

# parser = argparse.ArgumentParser(description='Prepared training dataset', formatter_class=argparse.RawTextHelpFormatter)
# parser.add_argument('--model_path', '-m', default='meta-llama/Llama-2-7b-chat-hf', help='the student model path')
# parser.add_argument('--feedback_path', '-fp', default='data/Llama-2-7b-chat-hf_gsm8k_feedback_round0.json',
#                     help='teacher feedbacks path')
# parser.add_argument('--rationale_path', '-rp', default='data/Llama-2-7b-chat-hf_gsm8k_rationale_round0.json',
#                     help='teacher rationales path')
# parser.add_argument('--seed', '-s', default=731, help='seed setting')
# parser.add_argument('--max_seq_length', '-ml', default=512, help='max_seq_length')
# parser.add_argument('--batch_size', '-bs', default=8, help="batch size")
# parser.add_argument('--teacher', default='mixed', choices=['mixed', 'gpt', 'gemini', 'mistral'],
#                     help="which model selected as teacher")
# args = parser.parse_args()

class LoadData():
    """
    load teacher_gsm8k_feedback_round0.json, teacher_gsm8k_rationale_round0.json
    return: huggingface DatasetDict{'train', 'valid', 'test'}
    Dataset: list[dict{'feedback':[str:LLM input, str:LLM output], 'rationale':[str:LLM input, str:LLM output]}]
    """

    def __init__(self, args):
        feedback_path = args.feedback_path
        rationale_path = args.rationale_path
        benchmark = feedback_path.split('_')[1]  # which benchmark
        seed = args.seed
        self.batch_size = args.batch_size
        with open(feedback_path, 'r') as f:
            feedbacks = json.load(f)
        with open(rationale_path, 'r') as f:
            rationales = json.load(f)
        assert len(feedbacks) == len(rationales)

        train_json = []
        self.seed_run(seed)  # seed everything
        feedback_input_temp = '\n'.join(teacher_model_feedback_prompt.split('\n')[2:])  # the input
        count = 0
        for i in range(len(rationales)):
            # instruction tuning of feedback
            cur_dict = {}
            try:
                if args.teacher == 'gpt':  # pick gpt response
                    cur_feedback = feedbacks[i]['gpt_feedback']
                    cur_rationale = rationales[i]['gpt_rationale'][0]
                elif args.teacher == 'gemini':  # pick gpt response
                    cur_feedback = feedbacks[i]['gemini_feedback']
                    cur_rationale = rationales[i]['gemini_rationale'][0]
                elif args.teacher == 'mistral':  # pick mistral response
                    cur_feedback = feedbacks[i]['mistral_feedback']
                    cur_rationale = rationales[i]['mistral_rationale'][0]
                else:
                    # random pick one feedback
                    cur_feedback = feedbacks[i][random.choice(['gpt_feedback', 'gemini_feedback', 'mistral_feedback'])]
                    # random pick one rationale, but sometimes there will be no correct rationale after peer-review
                    cur_rationale = random.choice(rationales[i]['correct_rationale'])

                # only take the samples with both feedback and rationale
                assert cur_feedback and cur_rationale

                # add training data about feedback and rationale
                if benchmark == 'strategyQA':
                    # feedback_input_temp: remove the first sentence from teacher_model_feedback_prompt
                    feedback_input = feedback_input_temp.format(feedbacks[i]['question'],
                                                                feedbacks[i]['student_wrong_pred'],
                                                                feedbacks[i]['gold_answer'].split('\t')[0].strip())
                    cur_dict['feedback'] = [train_feedback_prompt.format(feedback_input).strip(), cur_feedback.strip()]
                    cur_dict['rationale'] = [strategyQA_train_rationale_prompt.format(feedbacks[i]['question']).strip(),
                                             cur_rationale.strip()]
                elif benchmark == 'gsm8k':
                    feedback_input = feedback_input_temp.format(feedbacks[i]['question'],
                                                                feedbacks[i]['student_wrong_pred'],
                                                                feedbacks[i]['gold_answer'].split('####')[1].strip())
                    cur_dict['feedback'] = [train_feedback_prompt.format(feedback_input).strip(), cur_feedback.strip()]
                    cur_dict['rationale'] = [train_rationale_prompt.format(feedbacks[i]['question']).strip(),
                                             cur_rationale.strip()]

                elif benchmark == 'logiQA':
                    question = logiQA_student_answer_question_prompt.format(feedbacks[i]['context'],
                                                                            feedbacks[i]['question'],
                                                                            feedbacks[i]['options'])
                    feedback_input = feedback_input_temp.format(question,
                                                                feedbacks[i]['student_wrong_pred'],
                                                                feedbacks[i]['gold_answer'].strip())
                    cur_dict['feedback'] = [train_feedback_prompt.format(feedback_input).strip(), cur_feedback.strip()]
                    cur_dict['rationale'] = [logiQA_train_rationale_prompt.format(question).strip(),
                                             cur_rationale.strip()]

                elif benchmark == 'svamp':
                    feedback_input = feedback_input_temp.format(feedbacks[i]['question'],
                                                                feedbacks[i]['student_wrong_pred'],
                                                                feedbacks[i]['gold_answer'])
                    cur_dict['feedback'] = [train_feedback_prompt.format(feedback_input).strip(), cur_feedback.strip()]
                    cur_dict['rationale'] = [train_rationale_prompt.format(feedbacks[i]['question']).strip(),
                                             cur_rationale.strip()]

                count += 1
                train_json.append(cur_dict)
            except:
                continue
        print(
            f'For benchmark {benchmark}, original wrong predictions:{len(feedbacks)}, successfully processed {count} feedbacks and rationales with teacher as {args.teacher}!')

        with open(f'{benchmark}_instruction_tuning_round0_{args.teacher}.json', 'w') as f:
            json.dump(train_json, f, indent=4)
        hf_ds = load_dataset('json', data_files=f'{benchmark}_instruction_tuning_round0_{args.teacher}.json')

        self.hf_ds = DatasetDict({
            'train': hf_ds['train'],
            'valid': [],
            'test': []
        })
        print(f'training dataset: {len(self.hf_ds["train"])} samples, '
              f'validation dataset: {len(self.hf_ds["valid"])} samples.')

    def __call__(self, *args, **kwargs):
        return self.hf_ds

    def seed_run(self, seed=731):  # seed everything
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.use_deterministic_algorithms(True)


class CustomDataset(torch.utils.data.Dataset):
    """
    input: huggingface Dataset by class LoadData
    __getitem__ output: dict{"feedback":dict{"input_ids":list[max_seq_length],
    "labels":list[max_seq_length],"attention_mask":list[1*max_seq_length]},
    "rationale":dict{"input_ids":list[max_seq_length],
    "labels":list[max_seq_length],"attention_mask":list[max_seq_length]}}
    """

    def __init__(self, dataset, tokenizer, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.args = args

    def __getitem__(self, idx):
        feedback_instruction = self.tokenizer(self.dataset[idx]['feedback'][0])["input_ids"]
        rationale_instruction = self.tokenizer(self.dataset[idx]['rationale'][0])["input_ids"]
        pad_token_id = self.tokenizer.eos_token_id

        feedback_output = self.tokenizer(self.dataset[idx]['feedback'][1], add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        feedback = {}
        feedback["input_ids"] = feedback_instruction + feedback_output
        feedback["labels"] = [-100] * len(feedback_instruction) + feedback_output   # -100 for mask
        # padding to max_seq_length
        if len(feedback["input_ids"]) <= self.args.max_seq_length:
            feedback["input_ids"] = feedback["input_ids"] + [pad_token_id] * (
                        self.args.max_seq_length - len(feedback["input_ids"]))
            feedback["labels"] = feedback["labels"] + [pad_token_id] * (
                        self.args.max_seq_length - len(feedback["labels"]))
            feedback['attention_mask'] = [1] * len(feedback["input_ids"]) + [0] * (
                        self.args.max_seq_length - len(feedback["input_ids"]))
        else:
            feedback["input_ids"] = feedback["input_ids"][:self.args.max_seq_length]
            feedback["labels"] = feedback["labels"][:self.args.max_seq_length]
            feedback['attention_mask'] = [1] * self.args.max_seq_length

        rationale_output = self.tokenizer(self.dataset[idx]['rationale'][1], add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        rationale = {}
        rationale["input_ids"] = rationale_instruction + rationale_output
        rationale["labels"] = [-100] * len(rationale_instruction) + rationale_output
        # padding to max_seq_length
        if len(rationale["input_ids"]) <= self.args.max_seq_length:
            rationale["input_ids"] = rationale["input_ids"] + [pad_token_id] * (
                        self.args.max_seq_length - len(rationale["input_ids"]))
            rationale["labels"] = rationale["labels"] + [pad_token_id] * (
                        self.args.max_seq_length - len(rationale["labels"]))
            rationale['attention_mask'] = [1] * len(rationale["input_ids"]) + [0] * (
                        self.args.max_seq_length - len(rationale["input_ids"]))
        else:
            rationale["input_ids"] = rationale["input_ids"][:self.args.max_seq_length]
            rationale["labels"] = rationale["labels"][:self.args.max_seq_length]
            rationale['attention_mask'] = [1] * self.args.max_seq_length

        return feedback, rationale

    def __len__(self):
        return len(self.dataset)


class Collater():
    """
    input: batch of samples in CustomDataset, list[CustomDataset samples]
    output: dict{"feedback":dict{"input_ids":torch.tensor(batch_size*max_seq_length),
    "labels":torch.tensor(batch_size*max_seq_length),"attention_mask":torch.tensor(batch_size*max_seq_length)},
    "rationale":dict{"input_ids":torch.tensor(batch_size*max_seq_length),
    "labels":torch.tensor(batch_size*max_seq_length),"attention_mask":torch.tensor(batch_size*max_seq_length)}}
    """

    def __call__(self, data):
        feedbacks = defaultdict(list)
        rationales = defaultdict(list)
        for feedback, rationale in data:  # each sample in the batch
            for key, val in feedback.items():
                feedbacks[key].append(val)
            for key, val in rationale.items():
                rationales[key].append(val)
        # convert to tensors batch
        feedbacks = {key: torch.tensor(val) for key, val in feedbacks.items()}
        rationales = {key: torch.tensor(val) for key, val in rationales.items()}
        return {'feedback': feedbacks, 'rationale': rationales}


# Uncomment the following to run data.py individually

# d = LoadData(args)
# dataset = d()
# print(dataset)
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', cache_dir='local_models/')
# df = CustomDataset(dataset['train'], tokenizer, args)
# train_loader = DataLoader(df, shuffle=True, batch_size=4, collate_fn=Collater())
# for batch in train_loader:
#     # print(batch)
#     break
