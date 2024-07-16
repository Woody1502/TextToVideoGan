import cv2
import pandas as pd
import torch.nn as nn
import nltk
import glob
import os
from torchvision.io import read_video
import skvideo.io
import torchvision
from torchvision.io import VideoReader
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np
import torchvision.transforms as transforms
from torchsummary import summary
import torch
import gc
np.float = np.float64
np.int = np.int_
from nltk.corpus import stopwords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stop_words = set(stopwords.words('english'))
def categorize_words(text, wv_embeddings, device, stop_words):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    pos_tags = pos_tag(words)
    
    motion_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS'}
    content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'DT', 'IN', 'PRP', 'PRP$', 'WDT', 'WP', 'WP$', 'WRB'}
    
    motion = [word for word, tag in pos_tags if tag in motion_tags and word not in stop_words]
    content = [word for word, tag in pos_tags if tag in content_tags and word not in stop_words]
    
    motion_embeddings = np.array([wv_embeddings[word] for word in motion if wv_embeddings.has_index_for(word)])
    content_embeddings = np.array([wv_embeddings[word] for word in content if wv_embeddings.has_index_for(word)])
    
    embeddings_list1 = torch.tensor(motion_embeddings)
    embeddings_list2 = torch.tensor(content_embeddings)
    
    return torch.sum(embeddings_list1, dim=0).to(device), torch.sum(embeddings_list2, dim=0).to(device)


def list_files(directory):
    files = []
    for entry in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, entry)):
            files.append(entry)
    return files



def video_to_tensor(video_path, device):
    video=skvideo.io.vread('/home/alex/Рабочий стол/GAN/video/'+video_path)
    video_tensor = torch.tensor(video).to(device)
    #print('освобождение памяти')
    # Освобождение памяти
    #del video
    #torch.cuda.empty_cache()
    
    
    return video_tensor

# Пример использования

def display_video(video_array):
    # Проверяем, что видео массив не пустой
    if video_array is None or len(video_array) == 0:
        print("Ошибка: видео массив пустой.")
        return
    
    # Получаем количество кадров в видео
    num_frames = video_array.shape[0]
    
    for i in range(num_frames):
        # Получаем текущий кадр
        frame = video_array[i]
        
        # Отображаем кадр
        cv2.imshow('Video', frame)
        
        # Задержка для отображения кадров (например, 30 мс)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Закрываем все окна
    cv2.destroyAllWindows()
    
class Discriminator_I(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator_I, self).__init__()
        
        
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 6 x 6
            nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
class Discriminator_V(nn.Module):
    def __init__(self, nc=3, ndf=64, T=16, ngpu=1):
        super(Discriminator_V, self).__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x T x 96 x 96
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x T/2 x 48 x 48
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x T/4 x 24 x 24
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x T/8 x 12 x 12
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x T/16  x 6 x 6
            nn.Flatten(),
            nn.Linear(int((ndf*8)*(T/16)*6*6), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
class Generator_I(nn.Module):
    def __init__(self, nc=3, ngf=64, nz=60, ngpu=1):
        super(Generator_I, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 6 x 6
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 12 x 12
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 24 x 24
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 48 x 48
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 96 x 96
        )

    def forward(self, input):
        output = self.main(input)
        return output
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0, gpu=True):
        super(LSTM, self).__init__()

        output_size      = input_size
        self._gpu        = gpu
        self.hidden_size = hidden_size

        # define layers
        self.gru    = nn.LSTMCell(input_size, hidden_size)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn     = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, inputs, n_frames):
        '''
        inputs.shape()   => (batch_size, input_size)
        outputs.shape() => (seq_len, batch_size, output_size)
        '''
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [ self.bn(elm) for elm in outputs ]
        outputs = torch.stack(outputs)
        return outputs