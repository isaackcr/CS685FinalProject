from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch

def fine_tune(train_filename, test_filename, model_name="dmis-lab/biobert-base-cased-v1.2"):
    print("Load train dataset from disk: ", train_filename)
    train_dataset = load_from_disk(train_filename)
    print("Load test dataset from disk: ", test_filename)
    test_dataset = load_from_disk(test_filename)
    print("Load tokenizer from model: ", model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.model_max_length > 4096:
        print("- Tokenizer Model Max Length not found, setting to 512...")
        tokenizer.model_max_length = 512
    else:
        print("- Tokenizer Model Max was found: ", tokenizer.model_max_length)

    # tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            print("- Tokenizer Padding not found, setting to [PAD]...")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            print("- Tokenizer Padding not found, setting to eos_token: ", tokenizer.eos_token)
            tokenizer.pad_token = tokenizer.eos_token
    else:
        print("- Tokenizer Padding was found: ", tokenizer.pad_token)

    print("Tokenize training dataset...")
    train_dataset = train_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding='max_length'), batched=True)
    print("Tokenize test dataset...")
    test_dataset = test_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding='max_length'), batched=True)

    print("Load model: ", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # if model.model_max_length is None:
    #     print("- Tokenizer Model Max Length not found, setting to 512...")
    #     tokenizer.model_max_length = 512
    # else:
    #     print("- Tokenizer Model Max was found: ", tokenizer.model_max_length)
    #
    # # tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.pad_token is None:
    #     if tokenizer.eos_token is None:
    #         print("- Tokenizer Padding not found, setting to [PAD]...")
    #         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     else:
    #         print("- Tokenizer Padding not found, setting to eos_token: ", tokenizer.eos_token)
    #         tokenizer.pad_token = tokenizer.eos_token
    # else:
    #     print("- Tokenizer Padding was found: ", tokenizer.pad_token)
    #
    # model.config.pad_token_id = model.config.eos_token_id

    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    training_args = TrainingArguments(
        f"training_with_callbacks",
        evaluation_strategy=IntervalStrategy.EPOCH,  # "steps"
        save_strategy=IntervalStrategy.EPOCH,
        logging_steps=5,
        # eval_steps = 10, # Evaluation and Save happens every 50 steps
        save_total_limit=3,  # Only last 5 models are saved. Older ones are deleted.
        # learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        # weight_decay=0.01,
        push_to_hub=False,
        metric_for_best_model='f1'
        , load_best_model_at_end=True
        #     ,gradient_accumulation_steps=2
        #     ,gradient_checkpointing=True
        #     ,fp16=True
        , optim="adafactor")

    if torch.cuda.is_available():
        training_args.tf32 = True
        print('CUDA Available: ' + str(torch.cuda.is_available()))
        print('CUDA device_count: ' + str(torch.cuda.device_count()))
        print('CUDA current_device: ' + str(torch.cuda.current_device()))
        print('CUDA get_device_name[0]: ' + str(torch.cuda.get_device_name(0)))
    elif torch.backends.mps.is_available():
        training_args.use_mps_device=True
        print("Using MPS")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics)

    print("Train...")

    trainer.train()