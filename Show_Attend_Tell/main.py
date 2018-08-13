"""
Training

"""

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import argparse
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                            transform, args.batch_size, shuffle=True, num_workers = args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_dim).to(device)
    decoder = DecoderRNN(args.embed_dim, args.annot_dim, args.annot_num, args.hidden_dim, len(vocab), args.num_layers).to(device)

    # loss and optimizer setting
    criterion = nn.CrossEntropyLoss()
    num_epochs = args.num_epochs
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            #captions = pack_padded_sequence(captions[:, 1:], [l-1 for l in lengths], batch_first=True)[0]            
            #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
           
            outputs = decoder(features, captions[:, :-1], [l - 1 for l in lengths])
            outputs = pack_padded_sequence(outputs, [l-1 for l in lengths], batch_first=True)[0]
            captions = pack_padded_sequence(captions[:, 1:], [l-1 for l in lengths], batch_first=True)[0]
        
            #print("output", outputs.shape)
            #print("targets", captions.shape)
            loss = criterion(outputs, captions)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

    print
    'Number of epochs:', num_epochs
    for epoch in xrange(num_epochs):
        train(dataloader=dataloader, model=model, optimizer=optimizer, criterion=criterion,
              epoch=epoch, total_epoch=num_epochs)
        torch.save(model, './checkpoints/model_%d.pth' % (epoch))
        test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../cocoData/2014_images/train_resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='../cocoData/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_dim', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--annot_dim', type=int, default=512)
    parser.add_argument('--annot_num', type=int, default=196)
    parser.add_argument('--vocab_size', type=int, default=3000) 
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)




