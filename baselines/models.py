#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import *
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


max_len = 128
data_filepath = '/users/melodi/emetheni/word_order/data/frwac_jan/'

X_train, y_train = make_sets(data_filepath + 'train.tsv', max_len)
X_val, y_val = make_sets(data_filepath + 'dev.tsv', max_len)
X_test, y_test = make_sets(data_filepath + 'test.tsv', max_len)

print('made sets\n')

with open('vocab.json', 'r') as f:
    word2idx = json.load(f)
idx2word = {v: k for k, v in word2idx.items()}

# Load pretrained vectors
embeddings = load_pretrained_vectors(word2idx, "cc.fr.300.vec")
embeddings = torch.tensor(embeddings)

train_dataloader = data_loader(X_train, y_train, batch_size=50)
val_dataloader   = data_loader(X_val, y_val, batch_size=50)
test_dataloader  = data_loader(X_test, y_test, batch_size=50)



#########################  8. DEFINE THE LSTM MODEL  ##########################

class SentimentLSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p = 0.5):
        super().__init__()
        # params: "n_" means dimension
        self.n_vocab = n_vocab     # number of unique words in vocabulary
        self.n_layers = n_layers   # number of LSTM layers 
        self.n_hidden = n_hidden   # number of hidden nodes in LSTM
        
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first = True, dropout = drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words):
                                             # INPUT   :  (batch_size, seq_length)
        embedded_words = self.embedding(input_wfords)    # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)         # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden) # (batch_size*seq_length, n_hidden)
        fc_out = self.fc(lstm_out)                      # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)              # (batch_size*seq_length, n_output)
        sigmoid_out = sigmoid_out.view(batch_size, -1)  # (batch_size, seq_length*n_output)
        
        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1]               # (batch_size, 1)
        
        return sigmoid_last, h
    
    
    def init_hidden (self, batch_size):  # initialize hidden weights (h,c) to 0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return h


################  9. INSTANTIATE THE MODEL W/ HYPERPARAMETERS #################
n_vocab = len(word2idx)
n_embed = 300
n_hidden = 512
n_output = 1   # 1 ("positive") or 0 ("negative")
n_layers = 2

net = SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers,
                    pretrained_embedding=False,
#                     pretrained_embedding=embeddings,
                   )

#######################  10. DEFINE LOSS & OPTIMIZER  #########################
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

##########################  11. TRAIN THE NETWORK!  ###########################
print_every = 100
step = 0
n_epochs = 4  # validation loss increases from ~ epoch 3 or 4
clip = 5  # for gradient clip to prevent exploding gradient problem in LSTM/RNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Make sure that the weights in the embedding layer are not updated
net.word_embeddings.weight.requires_grad=False
embeddings = load_pretrained_vectors(word2idx, "cc.fr.300.vec")
embeddings = torch.tensor(embeddings)

for epoch in range(n_epochs):
    h = net.init_hidden(batch_size)
    
    for inputs, labels in train_dataloader:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        # making requires_grad = False for the latest set of h
        h = tuple([each.data for each in h])   
        
        net.zero_grad()
        output, h = net(inputs)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), clip)
        optimizer.step()
        
        if (step % print_every) == 0:            
            ######################
            ##### VALIDATION #####
            ######################
            net.eval()
            valid_losses = []
            v_h = net.init_hidden(batch_size)
            
            for v_inputs, v_labels in val_dataloader:
                v_inputs, v_labels = inputs.to(device), labels.to(device)
        
                v_h = tuple([each.data for each in v_h])
                
                v_output, v_h = net(v_inputs)
                v_loss = criterion(v_output.squeeze(), v_labels.float())
                valid_losses.append(v_loss.item())

            print("Epoch: {}/{}".format((epoch+1), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  "Validation Loss: {:.4f}".format(np.mean(valid_losses)))
            net.train()


################  12. TEST THE TRAINED MODEL ON THE TEST SET  #################
net.eval()
test_losses = []
num_correct = 0
test_h = net.init_hidden(batch_size)

for inputs, labels in test_loader:
    test_h = tuple([each.data for each in test_h])
    test_output, test_h = net(inputs, test_h)
    loss = criterion(test_output, labels)
    test_losses.append(loss.item())
    
    preds = torch.round(test_output.squeeze())
    correct_tensor = preds.eq(labels.float().view_as(preds))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)
    
print("Test Loss: {:.4f}".format(np.mean(test_losses)))
print("Test Accuracy: {:.2f}".format(num_correct/len(test_loader.dataset)))


###############################################################################
############  13. TEST THE TRAINED MODEL ON A RANDOM SINGLE REVIEW ############
###############################################################################
# def predict(net, review, seq_length = 200):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     words = preprocess(review)
#     encoded_words = [word2idx[word] for word in words]
#     padded_words = pad_text([encoded_words], seq_length)
#     padded_words = torch.from_numpy(padded_words).to(device)
    
#     if(len(padded_words) == 0):
#         "Your review must contain at least 1 word!"
#         return None
    
#     net.eval()
#     h = net.init_hidden(1)
#     output, h = net(padded_words, h)
#     pred = torch.round(output.squeeze())
# #     msg = "This is a positive review." if pred == 0 else "This is a negative review."
    
#     return msg


# review1 = "It made me cry."
# review2 = "It was so good it made me cry."
# review3 = "It's ok."
# review4 = "This movie had the best acting and the dialogue was so good. I loved it."
# review5 = "Garbage"
#                        ### OUTPUT ###
# predict(net, review1)  ## negative ##
# predict(net, review2)  ## positive ##
# predict(net, review3)  ## negative ##
# predict(net, review4)  ## positive ##
# predict(net, review5)  ## negative ##