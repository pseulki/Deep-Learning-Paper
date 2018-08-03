"""
'Show, Attend and Tell(https://arxiv.org/abs/1502.03044)'

Reference:
 - https://github.com/alecwangcq/show-attend-and-tell/
 - https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning
Modified some parts.
i
"""


import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        return x.cuda()

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained VGG19 with Batch Normalization without classifier layer."""
        super(EncoderCNN, self).__init__()
        vgg19 = models.vgg19_bn(pretrained=True)
        modules = list(vgg19.children())[0][:-1]  # Use Last CNN layer as feature extraction
        self.vgg = nn.Sequential(*modules)
        #self.linear = nn.Linear(100352, embed_size)
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.vgg(images) # Batch_size x 512 x 14 x 14
            features = features.view(features.size(0), features.size(1), -1) # Batch_size x 512 x 196
            features = features.transpose(1,2)
            #print("feature", features.shape)
        #features = features.reshape(features.size(0), -1)
        #features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):

    def __init__(self, embed_dim, annot_dim, annot_num, hidden_dim, vocab_size, num_layers=1, max_seq_length=20, dropout_ratio=0.5):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()

        self.embed_dim = embed_dim      #
        self.annot_dim = annot_dim      # annotation(feature) dimension: D
        self.annot_num = annot_num      # the number of annotations(features): L
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size    # vocabulary size : K
        self.num_layers = num_layers
        self.max_seg_length = max_seq_length    # length of the caption : C
        self.dropout_ratio = dropout_ratio

        self.embed = nn.Embedding(vocab_size, embed_dim) #[Batch_size, 26, 256]
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.lstm_cell = nn.LSTMCell(embed_dim + annot_dim, hidden_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Attention
        self.att_vw = nn.Linear(self.annot_dim, self.annot_dim, bias=False)
        self.att_hw = nn.Linear(self.hidden_dim, self.annot_dim, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(annot_num))
        self.att_w = nn.Linear(self.annot_dim, 1, bias=False)

    def _attention_layer(self, features, hiddens):
        """
        :param features:  batch_size  * 196 * 512
        :param hiddens:  batch_size * hidden_dim
        :return:
        """
        att_feats = self.att_vw(features) # N x L x D ? check
        #print("att_feats", att_feats.shape)
        att_h = self.att_hw(hiddens).unsqueeze(1)  # N x 1 x D ? check
        att_full = nn.ReLU()(att_feats + att_h + self.att_bias.view(1, -1, 1))
        #print("2", att_full.shape)
        att_out = self.att_w(att_full).squeeze(2)
        #print("3", att_out.shape)
        alpha = nn.Softmax(dim=1)(att_out) # N-L ? check
        context = torch.sum(features * alpha.unsqueeze(2), 1)

        return context, alpha

    def forward(self, features, captions, lengths):
        """
        :param features: batch_size * 196 * 512
        :param captions: batch_size * time_steps
        :param lengths:
        :return:
        """
        batch_size, time_step = captions.data.shape
        #print("batch_size", batch_size)
        #print(time_step)
        vocab_size = self.vocab_size
        #print("vocab_size", vocab_size)
        embed = self.embed
        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out
        #print("lstm input dim", self.embed_dim + self.annot_dim)
        word_embeddings = self.embed(captions)
        feats = torch.mean(features, 1)  # batch_size * 512
        h0, c0 = self.get_start_states(batch_size)

        predicts = to_var(torch.zeros(batch_size, time_step, vocab_size))
        for step in range(time_step):
            batch_size = sum(i >= step for i in lengths)
            if step != 0:
                feats, alpha = attention_layer(features[:batch_size, :], h0[:batch_size, :])
            words = (word_embeddings[:batch_size, step, :]).squeeze(1)
            inputs = torch.cat([feats, words], 1)
            #print("feats", feats.shape)
            #print("words", words.shape)
            #print("input", inputs.shape)
            h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = fc_out(fc_dropout(h0)) if fc_dropout is not None else fc_out(h0)
            predicts[:batch_size, step, :] = outputs

        return predicts

    def sample(self, feature, max_len=20):
        # greedy sample
        embed = self.embed
        lstm_cell = self.lstm_cell
        fc_out = self.fc_out
        attend = self._attention_layer
        batch_size = feature.size(0)

        sampled_ids = []
        alphas = [0]

        words = embed(to_var(torch.ones(batch_size, 1).long())).squeeze(1)
        h0, c0 = self.get_start_states(batch_size)
        feats = torch.mean(feature, 1) # convert to batch_size*512

        for step in range(max_len):
            if step != 0:
                feats, alpha = attend(feature, h0)
                alphas.append(alpha)
            inputs = torch.cat([feats, words], 1)
            h0, c0 = lstm_cell(inputs, (h0, c0))
            outputs = fc_out(h0)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.unsqueeze(1))
            words = embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze(), alphas

    def get_start_states(self, batch_size):
        hidden_dim = self.hidden_dim
        h0 = to_var(torch.zeros(batch_size, hidden_dim))
        c0 = to_var(torch.zeros(batch_size, hidden_dim))
        return h0, c0


if __name__ == '__main__':
    # for test
    annot_dim = 512
    annot_num = 196
    embed_dim = 512
    hidden_dim = 512
    vocab_size = 1000
    num_layers = 1
    dropout_ratio = 0.5
    model = DecoderRNN(embed_dim, annot_dim, annot_num, hidden_dim, vocab_size, num_layers, dropout_ratio)

    n_sample = 10
    features = to_var(torch.randn(n_sample, annot_num, annot_dim))
    caption = to_var(torch.zeros(n_sample, 20).long())
    lengths = [1, 2, 3, 2, 3, 2, 3, 20, 6, 4]

    model.train()
    model.cuda()

    for name, param in model.named_parameters():
        print (name, param.size())

    ss = model(features, caption, lengths)

