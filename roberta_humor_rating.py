import torch, argparse, pandas, numpy

from sklearn.metrics import accuracy_score, mean_squared_error


from hahadataset import HahaDataset

from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import TrainingArguments, Trainer

def metrics_acc(eval_pred):

    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc}

def metrics_rmse(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions
    rmse = mean_squared_error(labels,preds)

    rmse = numpy.float64(rmse)

    return{"rmse": rmse}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_from_checkpoint', type=str)
    parser.add_argument('--continue_training', type=str)
    parser.add_argument('--output_directory', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=4)
    args = parser.parse_args()
    task = 'humor_rating'

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_dataset = HahaDataset(input_file='data/train.csv', tokenizer=tokenizer, max_len=256, task=task,
                                split='train')
    eval_dataset = HahaDataset(input_file='data/train.csv', tokenizer=tokenizer, max_len=256, task=task,
                               split='eval')
    test_dataset = HahaDataset(input_file='data/public_dev.csv', tokenizer=tokenizer, max_len=256, task=task,
                               split='test')

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)

    warmup_steps = int(args.max_steps * .01)

    training_args = TrainingArguments(
        output_dir=args.output_directory,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        logging_steps=25,
        save_total_limit=1,
        evaluate_during_training=True,
        eval_steps=50,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_rmse',
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_rmse,
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    humor_rating_preds = [pred[0] for pred in predictions.predictions]

    output_list = []
    for pred in humor_rating_preds:
        temp = {}
        temp['humor_rating'] = pred
        output_list.append(temp)

    out_df = pandas.DataFrame(output_list)
    out_df.to_csv('submission_humor_rating.csv', index_label='id')
