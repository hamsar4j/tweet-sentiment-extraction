## NIE-MVI Semestral Project - Tweet Sentiment Extraction

### Author Name: S Hamsaraj

### Project inspired from: https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview

### Project Blog Link: https://medium.com/@hamsarajs8/tweet-sentiment-extraction-3761dfcc432c

In this project, my goal is to train and evaluate models that can accurately categorize tweets 
into three sentiment classes: positive, negative, and neutral. The objective is to achieve the
highest accuracy in sentiment classification.

### Files
- **train.csv** consists of the training data from Kaggle
- **test.csv** consists of the test data from Kaggle
- **RNN.ipynb** consists of the LSTM and GRU models that I created for training and testing the data
- **Transformers.ipynb** consists of the transformer models that I created for training and testing the data

### RNN.ipynb

#### import the necessary libraries for the RNN

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

#### load training data from csv file

`train_data = pd.read_csv('train.csv')`

#### load test data from csv file

`test_data = pd.read_csv('test.csv')`

#### drop rows with missing values in the **'text'** column

```python
train_data = train_data.dropna(subset=['text'])
test_data = test_data.dropna(subset=['text'])
```

#### extract text and sentiment columns
```python
x_train = train_data['text']
y_train = train_data['sentiment']
x_test = test_data['text']
y_test = test_data['sentiment']
```

#### process the data

I replaced all the URL links with 'http' for uniformity and also lower-cased all the characters in the text.

```python
def preprocess(data):
	modified_data = []
	for tweet in data:
		tweet_words = []
		for word in tweet.split():
			if word.startswith('http'):
				word = 'http'
			tweet_words.append(word)
		modified_tweet = " ".join(tweet_words)
		modified_tweet = modified_tweet.lower()
		modified_data.append(modified_tweet)
	return modified_data
x_train = preprocess(x_train)
```

#### a tokenizer is created and fitted on the text data from the training set.
The tokenizer will be used to convert text into sequences of numbers, in order 
for it to be processed by the LSTM model later

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text'])
```
#### text sequences from both training and test sets are converted into sequences of numbers using the previously fitted tokenizer.
```python
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
```
#### sequences are padded to ensure they have the same length
```python
x_train_padded = pad_sequences(x_train_seq)
x_test_padded = pad_sequences(x_test_seq)
```
#### convert data to tensors
```python
x_train_tensor = torch.tensor(x_train_padded).long()

x_test_tensor = torch.tensor(x_test_padded).long()
```
#### one-hot encoding - convert categorical data into a binary matrix - negative -> [1,0,0], neutral -> [0,1,0], positive -> [0,0,1]

Just like the text has to be converted into sequences of numbers, the labels also have to be converted. Here, we will be using one-hot encoding to convert categorical data into a binary matrix.

```python
y_train = pd.get_dummies(y_train, dtype=int).to_numpy()
y_test = pd.get_dummies(y_test, dtype=int).to_numpy()
```
#### create custom dataset class

As we are using our own dataset that has been modified, we will be creating our own custom dataset class.

```python
class MyDataset(Dataset):
	def __init__(self, data, target):
		self.x = data
		self.y = target

def __len__(self):
	return len(self.x)

def __getitem__(self, idx):
	return self.x[idx], self.y[idx]
```
#### create DataLoader for training and testing datasets

Next, we will create dataloaders for the training process. We will be using batch size = 8 and shuffling the training dataset.

```python
train_dataset = MyDataset(x_train_tensor, torch.tensor(y_train, dtype=torch.float32))

test_dataset = MyDataset(x_test_tensor, torch.tensor(y_test, dtype=torch.float32))
```
#### specify batch size for training and testing
```python
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

### Long Short Term Memory (LSTM) model

Now we will be creating the LSTM model using PyTorch, that consists of an embedding layer, LSTM layer and the final output layer. We will also be using a regularisation technique of dropout to prevent overfitting.

```python
class LSTM(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, dropout_rate):
		super(LSTM, self).__init__()
		# converts input indices to dense vectors of fixed size (hidden_size)
		self.embedding = nn.Embedding(input_size, hidden_size)
		# lstm layer
		self.lstm = nn.LSTM(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout_rate)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		# convert input indices to dense vectors
		embedded = self.embedding(x)
		# process the embedded sequence, change the order of dims for lstm layer
		lstm_out, _ = self.lstm(embedded.permute(1, 0, 2))
		# apply dropout to the output of the LSTM
		lstm_out = self.dropout(lstm_out[-1, :, :])
		# final output
		output = self.fc(lstm_out)
		return output
# Gated Recurrent Unit (GRU) model
# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(GRU, self).__init__()
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         gru_out, _ = self.gru(embedded.permute(1, 0, 2))
#         output = self.fc(gru_out[-1, :, :])
#         return output
```

I attempted using 2 different types of models, LSTM and GRU.

#### set device to cuda if GPU is available for utilisation or else cpu will be used

`device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`
#### dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to zero during training.

`dropout_rate = 0.1`
#### input size is the number of tokens

`input_size = len(tokenizer.word_index) + 1`
#### specify number of hidden layers

`hidden_size = 64`
#### output size is the labels for all the data

`output_size = len(set(train_data['sentiment']))`

#### set model to be used
`model = LSTM(input_size, hidden_size, output_size, dropout_rate)`
#### send model to previously specified device CPU/GPU
`model = model.to(device)`
#### set loss criterion
`criterion = nn.CrossEntropyLoss()`
#### set learning rate
`learning_rate = 0.001`
#### set type of optimizer
`optimizer = optim.Adam(model.parameters(), lr=learning_rate)`

#### training loop

Finally we will be training the model.

```python
num_epochs = 20
losses = []

for epoch in range(num_epochs):
	model.train()
	for inputs, labels in train_loader:
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs.long())
		loss = criterion(outputs, labels.argmax(dim=1))
		loss.backward()
		optimizer.step()
	losses.append(loss.item())
	print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
  ```
#### save model
```python
save_path = './rnnmodel.pth'
torch.save(model.state_dict(), save_path)
```
#### plot and visualise losses
`plt.plot(losses)`

### load saved model

`model.load_state_dict(torch.load(save_path))`
#### evaluation loop

With the trained model, we will now evaluate it using our test set.

```python
model.eval()
with torch.no_grad():
	total_correct = 0
	total_samples = 0
	for inputs, labels in test_loader:
		outputs = model(inputs.long())
		_, predicted = torch.max(outputs, 1)	
		total_samples += labels.size(0)
		total_correct += torch.eq(predicted, torch.argmax(labels, dim=1)).sum().item()
	accuracy = total_correct / total_samples
	print(f'Accuracy: {accuracy:.4f}')
```
At this point, I experimented with different parameter values such as learning rate, hidden size within layers, dropout rates and the other parameters. However, even after many attempts the average accuracy was around 68% which was quite low.

### Transformer.ipynb

I decided that I have to use a Transformer Model instead to achieve better results.

#### import the necessary libraries
```python
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
```

#### load training data from csv file

`train_data = pd.read_csv('train.csv')`

#### load test data from csv file

`test_data = pd.read_csv('test.csv')`

#### drop rows with missing values in the 'text' column

```python
train_data = train_data.dropna(subset=['text'])
test_data = test_data.dropna(subset=['text'])
```

#### extract text and sentiment columns
```python
x_train = train_data['text']
y_train = train_data['sentiment']
x_test = test_data['text']
y_test = test_data['sentiment']
```

#### process the data

```python
def preprocess(data):
	modified_data = []
	for tweet in data:
		tweet_words = []
		for word in tweet.split():
			if word.startswith('http'):
				word = 'http'
			tweet_words.append(word)
		modified_tweet = " ".join(tweet_words)
		modified_tweet = modified_tweet.lower()
		modified_data.append(modified_tweet)
	return modified_data
x_train = preprocess(x_train)
```
#### import pre-trained transformer models and tokenizers

```python
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

# model_checkpoint = 't5-small'
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# model_checkpoint = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

# model_checkpoint = 'bert-base-cased'
# tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

# model_checkpoint = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

model_checkpoint = 'roberta-base-openai-detector'
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
```
I experimented with the above pre-trained transformer models and tokenizer to see which model 
gave the best accuracy for my dataset. The 'roberta-base-openai-detector', which is roberta-base
fine-tuned by OpenAI on the outputs of the 1.5B-parameter GPT-2 model, gave the best accuracy. Hence,
I decided to fine-tune this pre-trained model to further improve its test validation accuracy.

#### the train and test data are tokenised

Next, I tokenised my text and encoded my labels as by the dictionary shown below.

```python
x_train_tokenized = tokenizer(x_train, padding=True, truncation=True, return_tensors='pt', max_length=128)
x_test_tokenized = tokenizer(x_test, padding=True, truncation=True, return_tensors='pt', max_length=128)
```
#### the sentiment labels are encoded in the training set as per the dictionary
```python
y_train_mapped = {'neutral': 0, 'negative': 1, 'positive': 2}
y_train_encoded = [y_train_mapped[sentiment] for sentiment in y_train]
```
#### custom dataset and loader is created 

Following which, I created my custom dataset and dataloader.

```python
class MyDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

train_dataset = MyDataset(
    input_ids=x_train_tokenized['input_ids'],
    attention_mask=x_train_tokenized['attention_mask'],
    labels=torch.tensor(y_train_encoded)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```
#### my custom fine-tuned model is created

```python
class MySentimentModel(nn.Module):
    def __init__(self, num_classes, pretrained_model_name='roberta-base-openai-detector'):
        super(MySentimentModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        x = self.fc1(pooled_output)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

#### custom model is initialised
```python
num_classes = 3
model = MySentimentModel(num_classes)
```

#### parameters are initialised

The specific parameters I used are shown below. In addition, I also added a learning rate scheduler
which reduces the learning rate by a factor (gamma) after a specified number iterations (step_size).

```python
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

model.to(device)

lr=3e-5
weight_decay=0.2

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
criterion = torch.nn.CrossEntropyLoss()
```

### training loop

```python
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_steps = len(train_loader)

    for step, batch in enumerate(train_loader):
      # Move batch to GPU
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=attention_mask)
        logits = outputs
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{total_steps}], Loss: {loss.item():.4f}')

    average_loss = total_loss / total_steps
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}')
    scheduler.step()
```

### evaluation loop
```python
import numpy as np
from sklearn.metrics import accuracy_score
# evaluate on test data after training
model.eval()
with torch.no_grad():
    x_test_tokenized = tokenizer(x_test, padding=True, truncation=True, return_tensors='pt', max_length=128)
    # move test data to GPU
    inputs_test = x_test_tokenized['input_ids'].to(device)
    attention_mask_test = x_test_tokenized['attention_mask'].to(device)

    outputs = model(inputs_test, attention_mask=attention_mask_test)
    predictions = torch.argmax(outputs, dim=1).tolist()

# calculate accuracy
true_labels = y_test.map({'neutral': 0, 'negative': 1, 'positive': 2})
predicted_labels = np.array(predictions)
accuracy = accuracy_score(true_labels, predicted_labels)

print(f'Accuracy: {accuracy * 100:.2f}%')
```
The final best accuracy after fine-tuning that I managed to obtain was about 80%, which beat the original pre-trained model
that came in at about 78%.