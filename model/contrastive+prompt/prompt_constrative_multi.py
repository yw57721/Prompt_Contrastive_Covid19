import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertForSequenceClassification, AutoModel, AutoTokenizer, BertTokenizer, BertForMaskedLM, AdamW, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import logging
import argparse
import torch
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
logging.basicConfig(level=logging.INFO)
num_class = {'2': 5, '3':5, '4':5, '5':4, '6':7, '7':10}
class LecCallTag():
    # Data process
    def data_process(self, data_file, qid, eda):
        if eda == 't':
            column_index = 'label'
        else:
            column_index = 'q'+qid+'_label'
        df = pd.read_csv(data_file, usecols=['text',column_index])
        df = df.drop(df[df[column_index]==-1].index)

        global class_weights
        class_weights = (1-(df[column_index].value_counts().sort_index() / len(df))).values
        class_weights = torch.from_numpy(class_weights).float().to("cuda")

        # train_data = np.array(df)
        # train_data_list = train_data.tolist()
        # df = pd.DataFrame(train_data_list)

        text = df['text'].tolist()
        label = df[column_index].tolist()
        return text, label
    
    # model, tokenizer
    def create_model_tokenizer(self, model_name, qid):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels = num_class[qid])
        return tokenizer, model
    
    # Construct dataset
    def create_dataset(self, text, label, tokenizer, max_len):
        X_train, X_temp, Y_train, Y_temp = train_test_split(text, label, test_size=0.3, random_state=1)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.6667, random_state=1)
        logging.info('train set: %s items, \nval set: %s itmes, \nteset set: %s itmes' %(len(X_train), len(X_val), len(X_test)))
        train_dict = {'text': X_train, 'labels': Y_train}
        val_dict = {'text': X_val, 'labels': Y_val}
        test_dict = {'text': X_test, 'labels': Y_test}
        train_dataset = Dataset.from_dict(train_dict)
        val_dataset = Dataset.from_dict(val_dict)
        test_dataset = Dataset.from_dict(test_dict)
        def preprocess_function(examples):
            text_token = tokenizer(examples['text'], padding='max_length',truncation=True, max_length=max_len)
            return text_token
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        val_dataset = val_dataset.map(preprocess_function, batched=True)
        test_dataset = test_dataset.map(preprocess_function, batched=True)
        return train_dataset, val_dataset, test_dataset
    
    # Construct trainer
    def create_trainer(self, model, train_dataset, val_dataset, checkpoint_dir, epoch, batch_size, tokenizer):
        class WeightedLossTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                outputs = model(**inputs)

                logits = outputs.get("logits")
                labels = inputs.get("labels")
                loss_func = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_func(logits, labels)
                return (loss, outputs) if return_outputs else loss
        args = TrainingArguments(
            checkpoint_dir,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch,
            weight_decay=0.01,
            metric_for_best_model='accuracy',
            load_best_model_at_end = True
        )
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            report = classification_report(labels, preds, digits=4)
            logging.info(report)
            
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        trainer = WeightedLossTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        return trainer

def test(test_dataset, trainer, qid):
    # Prediction
    raw_pred = trainer.predict(test_dataset)
    labels = raw_pred.label_ids
    preds = raw_pred.predictions.argmax(-1)

    cm = confusion_matrix(y_true=labels, y_pred=preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    # Save image
    plt.savefig('./results/constrastive/multiclass/'+'contrastive_multiclass_'+qid+'.png')

    report = classification_report(labels, preds, digits=4)
    my_open = open('./results/constrastive/multiclass/'+'contrastive_multiclass_'+qid+'.txt', 'w')
    my_open.write(report)
    my_open.close()

def main(model_name, qid, epoch, eda, t):
    lct = LecCallTag()
    if eda == 't':
        data_file = f'./data/eda/{qid}_mul.csv'
    else :
        data_file = './data/aggre_multiclass.csv'
    checkpoint_dir = "./results/checkpoint/"
    batch_size = 32
    max_len = 128
    
    text, label = lct.data_process(data_file, qid, eda)
    tokenizer, model = lct.create_model_tokenizer(model_name, qid)
    train_dataset, val_dataset, test_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    trainer = lct.create_trainer(model, train_dataset, val_dataset, checkpoint_dir, epoch, batch_size, tokenizer)
    if t == 'f':
        trainer.train()
    test(test_dataset, trainer, qid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--type', default='bt',
                    help='use different pretrained model to train and test')
    parser.add_argument('--qid', default='2')
    parser.add_argument('--ep', default='20', help='number of epochs')
    parser.add_argument('--eda', default='f', help='data argumentation or not')
    parser.add_argument('--test', default='f', help='test or not')
    args = parser.parse_args()

    if args.type == 'bt':
        model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    elif args.type == 'rb':
        model_name = "princeton-nlp/sup-simcse-roberta-base"
    else :
        model_name = args.type

    main(model_name, args.qid, int(args.ep), args.eda, args.test)
