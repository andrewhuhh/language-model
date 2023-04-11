import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.lower()

# Training data file (.txt format)
minishake = "training_data\\minishake.txt"
andrew = "training_data\\sampletxt.txt"
text = load_data(minishake)

# Create the character mapping dictionaries:
unique_chars = sorted(list(set(text)))
char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(unique_chars)}

# Define the RNN (recurrent neural net) model:
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

# Set up training parameters and the model:
input_size = len(unique_chars)
hidden_size = 512
output_size = len(unique_chars)
n_layers = 1
seq_length = 150
learning_rate = 0.005
batch_size = 64
n_epochs = 50

model = CharRNN(input_size, hidden_size, output_size, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the training loop:
def train(model, text, char_to_idx, idx_to_char, n_epochs, seq_length, batch_size, criterion, optimizer):
    for epoch in range(n_epochs):
        total_loss = 0
        for i in range(0, len(text) - seq_length * batch_size, seq_length * batch_size):
            inputs = torch.LongTensor([[char_to_idx[ch] for ch in text[i+j:i+j+seq_length]] for j in range(0, seq_length * batch_size, seq_length)])
            targets = torch.LongTensor([[char_to_idx[ch] for ch in text[i+j+1:i+j+seq_length+1]] for j in range(0, seq_length * batch_size, seq_length)])

            optimizer.zero_grad()
            hidden = model.init_hidden(batch_size)
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(batch_size * seq_length, -1), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / (len(text) // (seq_length * batch_size))}")

# Train the model:
train(model, text, char_to_idx, idx_to_char, n_epochs, seq_length, batch_size, criterion, optimizer)

# Generate text using trained model:
def generate_text(model, seed, char_to_idx, idx_to_char, length=100, temperature=1.0):
    model.eval()
    hidden = model.init_hidden(1)
    seed_tensor = torch.LongTensor([char_to_idx[ch] for ch in seed]).unsqueeze(0)
    generated_text = seed

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(seed_tensor, hidden)
            probs = nn.functional.softmax(output.squeeze() / temperature, dim=0).numpy()
            probs_flat = probs.ravel()  # Flatten the probability distribution
            probs_norm = probs_flat / np.sum(probs_flat)  # Normalize the probabilities
            next_idx = np.random.choice(len(probs_norm), p=probs_norm)

            # Error logging stuff (comment out if working properly)
            if next_idx not in idx_to_char:
                #print(f"KeyError: {next_idx}")
                #print("Probs array:", probs_norm)
                continue

            next_char = idx_to_char[next_idx]
            generated_text += next_char
            seed_tensor = torch.LongTensor([next_idx]).unsqueeze(0)

    return generated_text

# Generation defaults. Replace seed with your prompt for demo
seed = "one plus one equals"
generated_text = generate_text(model, seed, char_to_idx, idx_to_char, length=200, temperature=0.8)
print(generated_text)