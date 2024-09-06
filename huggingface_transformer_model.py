import torch
import transformers
from transformers import BitsAndBytesConfig
from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer

from datasets import load_dataset

class Finetune():
    def __init__(self,datasets:str) -> None:
        self.dataset =  load_dataset(datasets)
        self.tokenized_dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    load_in_4bit=True,
                    torch_dtype=torch.float16,)
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

        self.training_args = TrainingArguments(
                        output_dir="./llama-finetuned",
                        per_device_train_batch_size=4,
                        per_device_eval_batch_size=4,
                        num_train_epochs=3,
                        logging_dir="./logs",
                        save_steps=1000,
                        save_total_limit=2,
                        evaluation_strategy="steps",
                        eval_steps=500,
                        logging_steps=200,
                        learning_rate=2e-5,
                        fp16=True,  # Enable mixed precision for faster training
                        gradient_accumulation_steps=4,  # Useful if memory is limited
                        warmup_steps=100,
                    )
        self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.tokenized_dataset["train"],
                eval_dataset=self.tokenized_dataset["validation"],
                tokenizer=self.tokenizer,
            )
        
        self.trainer.train()

        self.model.save_pretrained("./llama-finetuned")
        self.tokenizer.save_pretrained("./llama-finetuned")

    def tokenize_function(self,example):
        return self.tokenizer(
            example["prompt"], 
            padding="max_length", 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
    

class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": True,
            },
        )
        self.terminators = [self.pipeline.tokenizer.eos_token_id]
  
    def get_response(
          self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9
      ):
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.construct_prompt(user_prompt)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators[0],  # Single EOS token id
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]
    
    def construct_prompt(self, message_history):
        prompt = ""
        for message in message_history:
            if message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n"
            elif message["role"] == "system":
                prompt += f"System: {message['content']}\n"
        return prompt

    def chatbot(self, system_instructions="you are friendly ai."):
        conversation = [{"role": "system", "content": system_instructions}]
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break
            response, conversation = self.get_response(user_input, conversation)
            print(f"Assistant: {response}")
  
if __name__ == "__main__":
    bot = Llama3("C:/Users/astro/Desktop/Python_Ai_Project/meta-llama/Meta-Llama-3-8B-Instruct")
    bot.chatbot()
