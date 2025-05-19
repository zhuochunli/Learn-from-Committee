# for strategyQA
strategyQA_student_answer_question_prompt = '''
Questions: {}
Let's think step by step. First, you should answer "true" or "false". Then, you should explain how you drew this conclusion.
'''

# for logiQA
logiQA_student_answer_question_prompt = '''
Given the context: {}
Query: {}
There are four options in the list: {}. 
Give me your answer and rationale immediately. Let's think step by step.
'''

teacher_model_feedback_prompt = '''
Imagine you are a teacher, I will give you one student's incorrect answer to a question. You should point out the mistakes in the student's answer.
Question: {}
Student's Answer: {}
Hint: The correct answer should be {}.
'''

teacher_model_peer_review_prompt = '''
Imagine you are a reviewer, I will give you one submission about the rationale for a question. You should simulate the peer-review process by evaluating the rationale based on its correctness and soundness. Let's think step by step, but your final answer should only be one number, ranging from 1-5 (the higher the score is, the more possible you think the rationale is correct). The score should be wrapped by square brackets, such as: [5].
Question: {}
Rationale: {}
Hint: The correct answer should be {}.
'''

train_feedback_prompt = '''
### Instruction:
Imagine you are a teacher, I will give you one student's incorrect answer to a question. You should point out the mistakes in the student's answer.
### Input:
{}
### Response:
'''

train_rationale_prompt = '''
### Instruction:
Answer the following question. Let's think step by step.
### Input:
{}
### Response:
'''

strategyQA_train_rationale_prompt = '''
### Instruction:
Answer the following question. Let's think step by step. First, you should answer "true" or "false". Then, you should explain how you drew this conclusion.
### Input:
{}
### Response:
'''

logiQA_train_rationale_prompt = '''
### Instruction:
Answer the following question based on the given context, query and options.
### Input:
{}
### Response:
'''