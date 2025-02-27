import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch:i for i ,ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])# decoder: take a list of integers, output a string

# Train and Test split
data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data, val_data = data[:n],data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()  # Disable gradient tracking (faster, uses less memory)
def estimate_loss():
    out = {}  # Dictionary to store average loss for train and val sets
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm, etc.)

    for split in ['train', 'val']:  # Loop over training and validation datasets
        losses = torch.zeros(eval_iters)  # Store multiple loss values for averaging
        
        for k in range(eval_iters):  # Run multiple evaluations for stable loss estimate
            X, Y = get_batch(split)  # Get a batch of data
            logits, loss = model(X, Y)  # Forward pass (get predictions and loss)
            losses[k] = loss.item()  # Store the loss value (convert tensor to float)
        
        out[split] = losses.mean()  # Compute and store the average loss
    
    model.train()  # Switch model back to training mode
    return out  # Return dictionary containing {"train": avg_loss, "val": avg_loss}

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # Get predictions for input idx
    
        if targets is None:                       # If no targets given (e.g., during inference)
            loss = None                           # No loss to calculate, set to None
        else:                                     # If targets are given (e.g., during training)
            B, T, C = logits.shape                # Get batch size (B), sequence length (T), vocab size (C)
            logits = logits.view(B*T, C)          # Flatten logits to (B*T, C) for loss function
            targets = targets.view(B*T)           # Flatten targets to (B*T,) to match logits
            loss = F.cross_entropy(logits, targets)  # Calculate cross-entropy loss between predictions and targets

        return logits, loss                       # Return predictions and loss (if computed)        

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range (max_new_tokens):
            # get the predictions
            logits , loss = self.forward(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] #becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = 1) #(B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)#(B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
     # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

      # sample a batch of data
    xb, yb = get_batch('train')

     # evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device = device)
print(decode(m.generate(context,max_new_tokens = 500)[0].tolist()))