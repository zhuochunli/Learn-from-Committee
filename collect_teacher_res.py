import os
from time import sleep
import openai
import google.generativeai as genai
from tqdm import tqdm
import json
import re
from all_prompts import teacher_model_feedback_prompt, teacher_model_peer_review_prompt, \
    strategyQA_student_answer_question_prompt, logiQA_student_answer_question_prompt
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser(description='Get correct rationales form teachers',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--gpt_api', help='the openAI api')
parser.add_argument('--mistral_api', help='the mistral api')
parser.add_argument('--gemini_api', help='the gemini api')
parser.add_argument('--student_wrong', default='data/Llama-2-7b-chat-hf_gsm8k_false_round0.json',
                    help='json file of student wrong predictions')
args = parser.parse_args()

gpt_key = os.environ.get('OPENAI_API_KEY', args.gpt_api)
mistral_key = os.environ.get('MISTRAL_API_KEY', args.mistral_api)
gemini_key = os.environ.get('GEMINI_API_KEY', args.gemini_api)

class TeacherLLMs:
    def __init__(self, args):
        self.dataset = args.student_wrong.split('_')[1]
        self.gpt_api = OpenAI(api_key=gpt_key)
        self.mistral_api = OpenAI(
            api_key=mistral_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        genai.configure(api_key=gemini_key)
        self.gemini_generation_config = {
            "temperature": 0.8,
        }
        # prevent gemini output None because of safety_rating
        self.gemini_safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

    def gpt_answer(self, prompt):
        try:
            response = self.gpt_api.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.8,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            output = response.choices[0].message.content
        except BaseException as E:
            print(E)
            output = None
        return output

    def gemini_answer(self, prompt):
        output = None
        gemini_model = genai.GenerativeModel('gemini-1.0-pro',
                                             generation_config=self.gemini_generation_config,
                                             safety_settings=self.gemini_safety_settings)
        try:
            for i in range(2):  # sometimes gemini will output nothing, so try again
                response = gemini_model.generate_content(prompt)
                for candidate in response.candidates:
                    output = [part.text for part in candidate.content.parts][0]  # get the first
                    break
                if output:
                    break
                # sleep(2)   # for rate limit
        except BaseException as E:
            print(E)
            output = None
        return output

    def mistral_answer(self, prompt):
        output = None
        try:
            chat_completion = self.mistral_api.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.8,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            output = chat_completion.choices[0].message.content
        except BaseException as E:
            print(E)
            output = None
        return output

    def collect_feedbacks(self, data):
        '''
        :param data: student_gsm8k_false_round0.json: list[dictionary{}]
                "id": ,
                "question": ques,
                "gold_answer": ration,
                "student_wrong_pred": pred,
                "student_wrong_final_answer": final_pred
        :return: feedback, list[dictionary{}]
                data.items(),
                "gpt_feedback": str,
                "gemini_feedback": str,
                "mistral_feedback": str,
        '''
        feedbacks = []
        for i in tqdm(range(len(data))):
            if self.dataset == 'logiQA':
                question = logiQA_student_answer_question_prompt.format(data[i]['context'], data[i]['question'],
                                                                        data[i]['options'])
                cur_prompt = teacher_model_feedback_prompt.format(question, data[i]['student_wrong_pred'],
                                                                  data[i]['gold_answer'])
            else:
                cur_prompt = teacher_model_feedback_prompt.format(data[i]['question'], data[i]['student_wrong_pred'],
                                                                  data[i]['gold_answer'])

            gpt_feedback = self.gpt_answer(cur_prompt)
            gemini_feedback = self.gemini_answer(cur_prompt)
            mistral_feedback = self.mistral_answer(cur_prompt)
            cur_dict = {'gpt_feedback': gpt_feedback, 'gemini_feedback': gemini_feedback,
                        'mistral_feedback': mistral_feedback}
            feedbacks.append(data[i] | cur_dict)
            # sleep(0.5)  # for rate limit of gemini

        return feedbacks

    def check_score(self, llm, data, rationale):
        '''
        check the score for {question, rationale} by llm int(1-5)
        '''
        if self.dataset == 'logiQA':
            question = logiQA_student_answer_question_prompt.format(data['context'], data['question'],
                                                                    data['options'])
            cur_prompt = teacher_model_peer_review_prompt.format(question, rationale,
                                                                 data['gold_answer'])
        else:
            cur_prompt = teacher_model_peer_review_prompt.format(data['question'], rationale, data['gold_answer'])

        if llm == 'gpt':
            output = self.gpt_answer(cur_prompt)
        elif llm == 'gemini':
            output = self.gemini_answer(cur_prompt)
        elif llm == 'mistral':
            output = self.mistral_answer(cur_prompt)
        else:
            raise NotImplementedError()

        try:
            score = re.findall(r'[[](.*?)[]]', output)  # score wrapped by []
            if score:
                score = score[-1]
                if '/' in score:  # e.g. [4/5]
                    score = int(score.split('/')[0])
                else:
                    score = int(re.findall(r'[0-9]+', score)[-1])  # find the last number
            else:
                score = int(re.findall(r'[0-9]+', output)[-1])  # find the last number
        except BaseException as E:
            # print(E)
            # print(llm)
            # print(output)
            score = 0
        return score

    def collect_correct_rationale(self, data):
        '''
        :param data: student_gsm8k_false_round0.json: list[dictionary{}]
                "id": ,
                "question": ques,
                "gold_answer": ration,
                "student_wrong_pred": pred,
                "student_wrong_final_answer": final_pred
        :return: feedback: list[dictionary{}]
                data.items(),
                "gpt_rationale":[rationale, score, score],
                "gemini_rationale":[rationale, score, score],
                "mistral_rationale":[rationale, score, score],
                "correct_rationale":[rationales]
        '''
        rationales = []

        for i in tqdm(range(len(data))):
            correct_rs = []
            if self.dataset == 'gsm8k' or self.dataset == 'svamp':
                gpt_r = self.gpt_answer(data[i]['question'])
                gemini_r = self.gemini_answer(data[i]['question'])
                mistral_r = self.mistral_answer(data[i]['question'])
            elif self.dataset == 'strategyQA':
                gpt_r = self.gpt_answer(strategyQA_student_answer_question_prompt.format(data[i]['question']))
                gemini_r = self.gemini_answer(strategyQA_student_answer_question_prompt.format(data[i]['question']))
                mistral_r = self.mistral_answer(strategyQA_student_answer_question_prompt.format(data[i]['question']))
            elif self.dataset == 'logiQA':
                gpt_r = self.gpt_answer(
                    logiQA_student_answer_question_prompt.format(data[i]['context'], data[i]['question'],
                                                                 data[i]['options']))
                gemini_r = self.gemini_answer(
                    logiQA_student_answer_question_prompt.format(data[i]['context'], data[i]['question'],
                                                                 data[i]['options']))
                mistral_r = self.mistral_answer(
                    logiQA_student_answer_question_prompt.format(data[i]['context'], data[i]['question'],
                                                                 data[i]['options']))

            # check each LLM's response by the other two LLMs, peer-review process
            gpt_r_gemini = self.check_score('gemini', data[i], gpt_r)
            gpt_r_mistral = self.check_score('mistral', data[i], gpt_r)
            total_score = []  # store valid scores
            if 0 < gpt_r_gemini < 6:
                total_score.append(gpt_r_gemini)
            if 0 < gpt_r_mistral < 6:
                total_score.append(gpt_r_mistral)
            if len(total_score) > 0 and sum(total_score) / len(total_score) > 3:  # only take rationale >= 4 points
                correct_rs.append(gpt_r)

            gemini_r_gpt = self.check_score('gpt', data[i], gemini_r)
            gemini_r_mistral = self.check_score('mistral', data[i], gemini_r)
            total_score = []  # store valid scores
            if 0 < gemini_r_gpt < 6:
                total_score.append(gemini_r_gpt)
            if 0 < gemini_r_mistral < 6:
                total_score.append(gemini_r_mistral)
            if len(total_score) > 0 and sum(total_score) / len(
                    total_score) > 3:
                correct_rs.append(gemini_r)

            mistral_r_gpt = self.check_score('gpt', data[i], mistral_r)
            mistral_r_gemini = self.check_score('gemini', data[i], mistral_r)
            total_score = []  # store valid scores
            if 0 < mistral_r_gpt < 6:
                total_score.append(mistral_r_gpt)
            if 0 < mistral_r_gemini < 6:
                total_score.append(mistral_r_gemini)
            if len(total_score) > 0 and sum(total_score) / len(
                    total_score) > 3:
                correct_rs.append(mistral_r)

            cur_dict = {'gpt_rationale': [gpt_r, gpt_r_gemini, gpt_r_mistral],
                        'gemini_rationale': [gemini_r, gemini_r_gpt, gemini_r_mistral],
                        'mistral_rationale': [mistral_r, mistral_r_gpt, mistral_r_gemini],
                        'correct_rationale': correct_rs}
            rationales.append(data[i] | cur_dict)

            # sleep(0.5)  # for rate limit of gemini

        return rationales


if __name__ == '__main__':
    teachers = TeacherLLMs(args)

    with open(args.student_wrong, 'r') as f:
        false_data = json.load(f)
    student_model = args.student_wrong.split('_')[0]

    with open(f'data/{student_model}_{teachers.dataset}_feedback_round0.json', 'w') as f:
        json.dump(teachers.collect_feedbacks(false_data), f, indent=4)
        print(f'saved {student_model}_{teachers.dataset}_feedback_round0.json !')

    with open(f'data/{student_model}_{teachers.dataset}_rationale_round0.json', 'w') as f:
        json.dump(teachers.collect_correct_rationale(false_data), f, indent=4)
        print(f'saved {student_model}_{teachers.dataset}_rationale_round0.json !')
