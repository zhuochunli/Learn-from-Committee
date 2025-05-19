import json
import torch
import argparse
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from data import LoadData, CustomDataset, Collater
from torch import nn

train_parser = argparse.ArgumentParser(description='train student model', formatter_class=argparse.RawTextHelpFormatter)
train_parser.add_argument('--model_path', '-m', default='meta-llama/Llama-2-7b-chat-hf', help='the student model path')
train_parser.add_argument('--flash_attention', '-fa', default=True, help='whether to use flash_attention')
train_parser.add_argument('--hf_token', '-ht', help='huggingface access token', required=True)
train_parser.add_argument('--output_path', '-op', default="checkpoints/llama2_gsm8k_round10_mixed", help='model output path')
train_parser.add_argument('--learning_rate', '-lr', default=1e-5, type=float, help='learning rate')
train_parser.add_argument('--max_seq_length', '-ml', default=512, type=int, help='max_seq_length')
train_parser.add_argument('--epochs', '-e', default=10, type=int, help="num_train_epochs")
train_parser.add_argument('--batch_size', '-bs', default=8, type=int, help="batch size")
train_parser.add_argument('--alpha', '-a', default=0.5, type=float, help="alpha number")
train_parser.add_argument('--feedback_path', '-fp', default='data/Llama-2-7b-chat-hf_gsm8k_feedback_round0.json', help='teacher feedbacks path')
train_parser.add_argument('--rationale_path', '-rp', default='data/Llama-2-7b-chat-hf_gsm8k_rationale_round0.json', help='teacher rationales path')
train_parser.add_argument('--teacher', default='mixed', choices=['mixed', 'gpt', 'gemini', 'mistral'], help="which model selected as teacher")
train_parser.add_argument('--seed', '-s', default=731, help='seed setting')
args = train_parser.parse_args()


class CustomTrainer:
    def __init__(self, args):
        self.args = args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_flash_attention_2=args.flash_attention,
            attn_implementation="flash_attention_2" if args.flash_attention else "sdpa",
            device_map="auto",
            cache_dir='local_models/',
        )
        # model.config.pretraining_tp = 1
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.hf_token)

        ld = LoadData(args)
        dataset = ld()
        train_ds = CustomDataset(dataset['train'], tokenizer, args)
        train_args = TrainingArguments(
            output_dir=args.output_path,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size if args.flash_attention else args.batch_size//2,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=50,
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            weight_decay=0.01,   # L2-regularization
            lr_scheduler_type="constant",
            seed=args.seed,
            save_total_limit=2,    # best one and last one
            # disable_tqdm=True
        )

        self.trainer = CustomSFTTrainer(args.alpha,
            model=model,
            train_dataset=train_ds,
            data_collator=Collater(),       # data.py Collater()
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            packing=True,
            args=train_args,
        )

    def train(self):
        # train
        self.trainer.train()
        print("Training completed!")
        # save model
        self.trainer.save_model()
        print("Trained model saved to ", self.args.output_path)


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, alpha, *args, **kwargs):
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.alpha = alpha  # decide the joint-learning loss weight

    def compute_hf_original_loss(self, model, inputs):
        return super().compute_loss(model, inputs, False)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_feedback = self.compute_hf_original_loss(model, inputs['feedback'])
        loss_rationale = self.compute_hf_original_loss(model, inputs['rationale'])
        total_loss = self.alpha * loss_feedback + (1-self.alpha) * loss_rationale
        # print(total_loss)
        return total_loss.mean()

    
if __name__ == '__main__':
    trainer = CustomTrainer(args)
    trainer.train()