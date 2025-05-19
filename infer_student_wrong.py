import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from all_prompts import strategyQA_student_answer_question_prompt, logiQA_student_answer_question_prompt
from datasets import load_dataset
import re
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get wrong predictions for student model', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model', '-m', default='meta-llama/Llama-2-7b-chat-hf', help='the student model path')
parser.add_argument('--dataset', default='strategyQA', choices=['gsm8k', 'svamp', 'strategyQA', 'logiQA'], help="which dataset")
parser.add_argument('--hf_token', '-ht', help='huggingface access token', required=True)
args = parser.parse_args()

# global LLM settings
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def split_sample(sample, dataset='gsm8k'):
    '''
    split question, answer and rationale for dataset GSM8K
    :param sample: dict{}
    :return: ques, ration, ans
    '''
    if dataset == 'gsm8k':
        ques = sample['question'].strip()
        ration = sample['answer'].strip()
        final_ans = sample['answer'].split('####')[1].strip()
    elif dataset == 'strategyQA':
        ques = sample['question'].strip()
        ration = sample['facts'].strip()
        final_ans = str(sample['answer']).strip().lower()
        ration = final_ans + '.\t' + ration      # rationale should also include the final answer
    elif dataset == 'logiQA':
        ques = sample['query'].strip()
        context = sample['context'].strip()
        options = sample['options']
        final_ans = str(sample['correct_option']).strip()
        return context, ques, options, final_ans
    elif dataset == 'svamp':
        ques = sample['question_concat'].strip()
        ration = sample['Equation'].strip()
        final_ans = sample['Answer'].strip()

    return ques, ration, final_ans


def cleanup(pred, dataset='gsm8k', options = None):
    """
    :param pred: generated text
    :param dataset: task
    :return: [cleaned_text, final_prediction]
    
    options: only deal with logiQA dataset
    """
    if dataset == 'gsm8k' or dataset == 'svamp':
        pred = pred.strip()
        temp = pred

        struct_ans_flag = False
        for answer_prefix in ['\nAnswer', 'Therefore, the answer is']:
            if answer_prefix in pred:
                temp = pred.split(answer_prefix)[1].strip()
                struct_ans_flag = True
                break

        # extract all numbers in prediction
        temp_ori = [item for item in re.findall(r'-?\d+\.?\$?,?\d*', temp)]
        temp = [item.strip('.') for item in re.findall(r'-?\d+\.?\d*', temp.replace(',', ''))]

        if len(temp) == 0:
            final_pred = 'ABSOLUTE_WRONG_FINAL_ANS'
            if struct_ans_flag:
                answer_prefix_idx = pred.index(answer_prefix)
                next_word = pred[answer_prefix_idx + len(answer_prefix):].split()
                if next_word[0] == ':':
                    if len(next_word) == 1:
                        next_word = ' '
                    else:
                        next_word = ': ' + next_word[1]
                else:
                    next_word = ' ' + next_word[0]
                pred = pred[:answer_prefix_idx + len(answer_prefix)] + next_word

        elif struct_ans_flag:
            final_pred = temp[0]
            answer_prefix_idx = pred.index(answer_prefix)
            if final_pred in pred[answer_prefix_idx:]:
                temp_idx = pred[answer_prefix_idx:].index(final_pred)
                pred = pred[:answer_prefix_idx + temp_idx + len(final_pred)]
            else:
                next_word = pred[answer_prefix_idx + len(answer_prefix):].split()
                if next_word[0] == ':':
                    next_word = ': ' + next_word[1]
                else:
                    next_word = ' ' + next_word[0]
                pred = pred[:answer_prefix_idx + len(answer_prefix)] + next_word

        elif not struct_ans_flag:
            final_pred = temp[-1]       # the last number
            if final_pred in pred:
                pred = pred[:pred.index(final_pred) + len(final_pred)]
            elif temp_ori[-1] in pred:
                pred = pred[:pred.index(temp_ori[-1]) + len(temp_ori[-1])]
            else:
                pass
        else:
            raise RuntimeError()

    elif dataset=='strategyQA':
        if len(pred) == 0:
            final_pred = 'ABSOLUTE_WRONG_FINAL_ANS'
        elif "true" in pred.lower():
            final_pred = 'true'
        elif "false" in pred.lower():
            final_pred = 'false'
        else:
            final_pred = 'ABSOLUTE_WRONG_FINAL_ANS'

    elif dataset == 'logiQA':
        final_pred = None
        # For the answer like :
        # Answer:
        #
        # The correct answer is: 'Some people who like peppers are southerners.'
        #
        # Rationale:
        try:
            ans = re.search(r'(.*)Answer:(.*)Rationale:(.*)', pred, flags=re.DOTALL)
            cur_pred = ans.group(2)
            for k in range(len(options)):   # find the corresponding option number
                if options[k] in cur_pred:
                    final_pred = str(k)
                    break
                final_pred = 'ABSOLUTE_WRONG_FINAL_ANS'
        except:
            final_pred = 'ABSOLUTE_WRONG_FINAL_ANS'

    return pred, final_pred


def predict(args):
    student_model = args.model
    res_wrongs = []
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir='local_models/',
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(student_model, token=args.hf_token)

    if args.dataset == 'strategyQA':
        dataset = load_dataset('ChilleD/StrategyQA', cache_dir='local_dataset/')
    elif args.dataset == 'logiQA':
        dataset = load_dataset("lucasmccabe/logiqa", cache_dir='local_dataset/')
    elif args.dataset == 'svamp':
        dataset = load_dataset("ChilleD/SVAMP", cache_dir='local_dataset/')
    else:
        dataset = load_dataset(args.dataset, 'main', cache_dir='local_dataset/')
    count = 0       # count the number of all wrong predictions
    for i in tqdm(range(len(dataset['train']))):
        sample = dataset['train'][i]
        if args.dataset == 'gsm8k' or args.dataset == 'svamp':
            ques, ration, final_ans = split_sample(sample, args.dataset)
            cur_prompt = ques
        elif args.dataset == 'strategyQA':
            ques, ration, final_ans = split_sample(sample, args.dataset)
            cur_prompt = strategyQA_student_answer_question_prompt.format(ques)
        elif args.dataset == 'logiQA':
            context, ques, options, final_ans = split_sample(sample, args.dataset)
            cur_prompt = logiQA_student_answer_question_prompt.format(context, ques, options)

        input_ids = tokenizer(cur_prompt, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=512,
            do_sample=True, top_p=0.9,
            temperature=0.3, eos_token_id=tokenizer.eos_token_id)
        student_ans = (tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(cur_prompt):])
        
        if args.dataset == 'logiQA':
            pred, final_pred = cleanup(student_ans, args.dataset, options)
        else:
            pred, final_pred = cleanup(student_ans, args.dataset)
        # print(pred)
        # print(options)
        # print(final_pred)
        # print(final_ans)
        if final_pred != final_ans:
            if args.dataset == 'logiQA':
                res_wrongs.append({
                    "id": i,
                    "context": context,
                    "question": ques,
                    "options": options,
                    "gold_answer": final_ans,
                    "student_wrong_pred": pred,
                    "student_wrong_final_answer": final_pred
                })
            elif args.dataset == 'svamp':
                res_wrongs.append({
                    "id": i,
                    "question": ques,
                    "gold_answer": final_ans,
                    "student_wrong_pred": pred,
                    "student_wrong_final_answer": final_pred
                })
            else:
                res_wrongs.append({
                    "id": i,
                    "question": ques,
                    "gold_answer": ration,
                    "student_wrong_pred": pred,
                    "student_wrong_final_answer": final_pred
                })
            count += 1
    
    with open(f"data/{student_model.split('/')[1]}_{args.dataset}_false_round0.json", 'w') as f:
        json.dump(res_wrongs, f, indent=4)

    print(f"For dataset {args.dataset}, infer {student_model.split('/')[1]} wrong answers completed! The number of wrong predictions: ", count, 'The Acc rate: ', 1-count/len(dataset['train']))


if __name__ == '__main__':
    predict(args)


