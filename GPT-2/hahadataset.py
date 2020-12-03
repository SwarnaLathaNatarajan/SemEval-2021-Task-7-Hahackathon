import pandas, torch, csv, random


class HahaDataset(torch.utils.data.Dataset):
    def __init__(self, input_file, tokenizer, max_len, task, split):

        self.input_file = input_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
        self.split = split

        self.examples = self.read_file(input_file)

        cutoff = int(len(self.examples) * .8)

        if split != 'test':
            random.seed(42)
            #random.shuffle(self.examples)

            if split == 'train':
                self.examples = self.examples[:cutoff]
            elif split == 'eval':
                self.examples = self.examples[cutoff:]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        instance = self.examples[idx]
        if self.split == 'test':

            sent = instance
            enc = self.tokenizer(
                sent,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_token_type_ids=True,
                return_tensors='pt'
            )

            return {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'token_type_ids': enc['token_type_ids'].squeeze(0),
            }

        sent = instance[0]
        if self.task == 'is_humor':
            label = int(instance[1])
        elif self.task == 'humor_rating' or self.task == "offense_rating":
            label = float(instance[1])

        enc = self.tokenizer(
            sent,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'labels': label
        }


    def read_file(self, file):
        inps = []

        with open(file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if self.split == 'test':
                    inps.append(line[1])
                elif self.task == 'humor_rating':
                    if line[3]:
                        inps.append((line[1], line[3]))
                elif self.task == 'offense_rating':
                    inps.append((line[1], line[5]))

        return inps[1:]
