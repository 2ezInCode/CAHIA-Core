from scipy.fftpack import fft, dct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.init as init
import pickle
import json, os
import argparse
import config_file
import random
from PIL import Image
from cn_clip.clip import load_from_name
from scipy import sparse
import dgl

from dgl.nn.pytorch import GATConv
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="weibo2")
parser.add_argument("-g", "--gpu_id", type=str, default="0")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = load_from_name("RN50", device=device, download_root='./')
model_clip.eval()

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):

    def __init__(
        self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()

        # 这里得把哪个数转为tensor
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        h = h.to(torch.float32)
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))

        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        ans = self.semantic_attention(semantic_embeddings)  # (N, D * K)
        return ans

class HAN(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)


def process_config(config):
    for k, v in config.items():
        config[k] = v[0]
    return config


def process_dct_img(img):
    img = img.numpy()  # size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    N = 8
    step = int(height / N)  # 28

    dct_img = np.zeros((1, 128, step * step, 1), dtype=np.float32)  # [1,64,784,1]
    fft_img = np.zeros((1, 128, step * step, 1))

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row + step), col:(col + step)], dtype=np.float32)
            # print('block:{}'.format(block.shape))
            block1 = block.reshape(-1, step * step, 1)  # [batch_size,784,1]
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]

            i += 1

    # for i in range(64):
    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real  # [batch_size,64, 784,1]

    fft_img = torch.from_numpy(fft_img).float()  # [batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250, 1])  # [batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1)  # torch.size = [64, 250]

    return new_img

def Conv1(in_channel,out_channel,kernel_size,stride,padding):
    return nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU())

class InceptionResNetA(nn.Module):
    def __init__(self,in_channel,scale=0.1):
        super(InceptionResNetA, self).__init__()
        self.branch1=Conv1(in_channel=in_channel,out_channel=32,kernel_size=1,stride=1,padding=0)
        self.branch2_1=Conv1(in_channel=in_channel,out_channel=32,kernel_size=1,stride=1,padding=0)
        self.branch2_2=Conv1(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1)
        self.branch3_1=Conv1(in_channel=in_channel,out_channel=32,kernel_size=1,stride=1,padding=0)
        self.branch3_2=Conv1(in_channel=32,out_channel=48,kernel_size=3,stride=1,padding=1)
        self.branch3_3=Conv1(in_channel=48,out_channel=64,kernel_size=3,stride=1,padding=1)
        self.linear=Conv1(in_channel=128,out_channel=384,kernel_size=1,stride=1,padding=0)
        self.out_channel=384
        self.scale=scale

        self.shortcut=nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=self.out_channel,kernel_size=1,stride=1,padding=0),
            )

    def forward(self,x):
        output1=self.branch1(x)
        output2=self.branch2_2(self.branch2_1(x))
        output3=self.branch3_3(self.branch3_2(self.branch3_1(x)))
        out=torch.cat((output1,output2,output3),dim=1)
        out=self.linear(out)
        x=self.shortcut(x)
        out=x+self.scale*out
        out=F.relu(out)
        return out

def ConvBNRelu2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class DctStem(nn.Module):
    def __init__(self, kernel_sizes, num_channels):
        super(DctStem, self).__init__()
        self.convs = nn.Sequential(
            ConvBNRelu2d(in_channels=1,
                         out_channels=num_channels[0],
                         kernel_size=kernel_sizes[0]),
            ConvBNRelu2d(
                in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=kernel_sizes[1],
            ),
            ConvBNRelu2d(
                in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=kernel_sizes[2],
            ),
            nn.MaxPool2d((1, 2)),
        )

    def forward(self, dct_img):
        x = dct_img.unsqueeze(1)
        img = self.convs(x)
        img = img.permute(0, 2, 1, 3)

        return img

class DctCNN(nn.Module):
    def __init__(self,
                 dropout,
                 kernel_sizes,
                 num_channels,
                 in_channel=128,
                 out_channels=64):
        super(DctCNN, self).__init__()

        self.stem = DctStem(kernel_sizes, num_channels)

        self.InceptionBlock = InceptionResNetA(
            in_channel,
            0.1
        )

        self.maxPool = nn.MaxPool2d((1, 122))

        self.dropout = nn.Dropout(dropout)

        self.conv = ConvBNRelu2d(128,
                                 out_channels,
                                 kernel_size=1)

    def forward(self, dct_img):
        dct_f = self.stem(dct_img)
        x = self.InceptionBlock(dct_f)
        x = self.maxPool(x)
        x = x.permute(0, 2, 1, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(-1)

        x = x.reshape(-1, 4096)

        return x

class PGD(object):

    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):

        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att

    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output

    def forward(self, Q, K, V):
        V_att = self.multi_head_attention(Q, K, V)
        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self):
        pass

    def cahia(self, x_tid, x_text, y, loss, i, total, params, pgd_word,g2):
        self.optimizer.zero_grad()

        # 这个forward上加了注解 表示功能由子类实现
        logit_defense, dist = self.forward(x_tid, x_text, g2)

        loss_classification = loss(logit_defense, y)
        loss_mse = nn.MSELoss()
        loss_dis = loss_mse(dist[0], dist[1])

        loss_defense =  loss_classification + loss_dis # 不加是9390 加了呢
        loss_defense.backward()

        K = 3
        pgd_word.backup_grad()
        for t in range(K):
            pgd_word.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_word.restore_grad()
            loss_adv, dist = self.forward(x_tid, x_text, g2)
            loss_adv = loss(loss_adv, y)
            loss_adv.backward()
        pgd_word.restore()

        self.optimizer.step()
        corrects = (torch.max(logit_defense, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                           loss_defense.item(),
                                                                           accuracy,
                                                                           corrects,
                                                                           y.size(0)))

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev,g2):

        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, weight_decay=0)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss = nn.CrossEntropyLoss()
        params = [(name, param) for name, param in self.named_parameters()]

        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])

            self.train()

            for i, data in enumerate(dataloader):
                total = len(dataloader)
                batch_x_tid, batch_x_text, batch_y = (item.cuda(device=self.device) for item in data)

                self.cahia(batch_x_tid, batch_x_text, batch_y, loss, i, total, params, pgd_word,g2)
                # 确保梯度的范数不超过指定的最大范数
                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

            self.evaluate(X_dev_tid, X_dev, y_dev,g2)

    def evaluate(self, X_dev_tid, X_dev, y_dev,g2):
        y_pred = self.predict(X_dev_tid, X_dev, g2)
        acc = accuracy_score(y_dev, y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("save model!!!   at ", self.config['save_path'])


    def predict(self, X_test_tid, X_test, g2):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)
                logits, dist = self.forward(batch_x_tid, batch_x_text,g2)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred


class resnet50():
    def __init__(self):
        self.newid2imgnum = config['newid2imgnum']
        self.model = models.resnet50(pretrained=True).cuda()
        # 注意是300
        self.model.fc = nn.Linear(2048, 300).cuda()
        torch.nn.init.eye_(self.model.fc.weight)
        self.path = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_images/weibo_images_all/'
        self.trans = self.img_trans()

    def img_trans(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    def forward(self, xtid):
        img_list = []
        for newid in xtid.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            imgpath = self.path + imgnum + '.jpg'
            im = np.array(self.trans(Image.open(imgpath)))
            im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
            img_list.append(im)
        batch_img = torch.cat(img_list, dim=0).cuda()

        img_output = self.model(batch_img)
        return img_output, batch_img


class frequencyCNN123(NeuralNetwork):
    def __init__(self):
        super(frequencyCNN123, self).__init__()
        self.newid2imgnum = config['newid2imgnum']
        self.path = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_images/weibo_images_all/'
        self.trans = self.img_trans()

        self.dct_img = DctCNN(dropout=0.5,
                              kernel_sizes=[3, 3, 3],
                              num_channels=[32, 64, 128],
                              in_channel=128,
                              out_channels=64).cuda()
        self.linear_dct = nn.Linear(4096, 300).cuda()
        self.bn_dct = nn.BatchNorm1d(300).cuda()
        self.dropout = nn.Dropout(0.5).cuda()
        self.drop_and_BN = 'drop-BN'

    def drop_BN_layer(self, x, part='dct'):
        if part == 'dct':
            bn = self.bn_dct

        if self.drop_and_BN == 'drop-BN':
            x = self.dropout(x)
            x = bn(x)
        return x

    def img_trans(self):
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor()
             ])
        return transform

    def forward(self, xtid):
        img_list = []
        for newid in xtid.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            imgpath = self.path + imgnum + '.jpg'
            im = Image.open(imgpath)
            im = self.trans(im.convert('L'))
            im = process_dct_img(im)
            img_list.append(im)


        batch_img = torch.cat(img_list, dim=0).view(len(img_list), 128, 250).cuda()

        dct_out = self.dct_img(batch_img)  # 2 *4096 的tensor
        dct_out = F.relu(self.linear_dct(dct_out))  # 2*256 的tensor
        dct_out = self.drop_BN_layer(dct_out, part='dct')  # 2*256 的tensor
        return dct_out


class CAHIA(NeuralNetwork):
    def __init__(self, config, adj, g2):
        super(CAHIA, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape  # D:300 V:17207
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.HANuse = HAN(len(g2), 300, 8, 300, [8],0.6).to(device)

        self.mh_attention = TransformerBlock(input_size=300, n_heads=8, attn_dropout=0)

        self.image_embedding = resnet50()
        self.fre_embedding = frequencyCNN123()

        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))

        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])
        self.dropout = nn.Dropout(dropout_rate)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.fc3 = nn.Linear(1800, 900)
        self.fc4 = nn.Linear(900, 600)
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=config['num_classes'])

        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.init_weight()


    # 初始化全连接层的权重
    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)

    def forward(self, X_tid, X_text,g2):

        iembedding, batch_img = self.image_embedding.forward(X_tid)

        with torch.no_grad():
            zero_tensor = torch.zeros(X_text.shape[0], 2).to(device)
            X_text1 = torch.cat([X_text, zero_tensor], dim=1).long().to(device)
            logits_per_image, logits_per_text = model_clip.get_similarity(batch_img, X_text1)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()  #
            sum_probs = np.trace(probs)


        X_text = self.word_embedding(X_text)
        if self.config['user_self_attention'] == True:
            X_text = self.mh_attention(X_text, X_text, X_text)

        # TODO HAN
        node_embedding = config['node_embedding'] # 6963 * 300 作为我们的h 节点表征
        node_embedding = torch.from_numpy(node_embedding)
        node_embedding = node_embedding.to(device)
        rembedding = self.HANuse(g2, node_embedding)
        rembedding = rembedding[X_tid]

        X_text = X_text.permute(0, 2, 1)
        conv_block = [rembedding]
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)

        # 将关系嵌入和x-test拼起来 得到卷积特征 并取出图特征 和 文本特征
        conv_feature = torch.cat(conv_block, dim=1)
        graph_feature, text_feature = conv_feature[:, :300], conv_feature[:, 300:]
        bsz = text_feature.size()[0]

        # 这里频因为换了resnetv2，所以得mean一下
        fembedding = self.fre_embedding.forward(X_tid)
        fembedding = torch.mean(fembedding.view(bsz, -1, 300), dim=1)

        img_enhanced = self.mh_attention(iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300), \
                                         fembedding.view(bsz, -1, 300))

        # 文自注意力
        self_att_t = self.mh_attention(text_feature.view(bsz, -1, 300), text_feature.view(bsz, -1, 300), \
                                       text_feature.view(bsz, -1, 300))
        # 图网络自注意力
        self_att_g = self.mh_attention(graph_feature.view(bsz, -1, 300), graph_feature.view(bsz, -1, 300), \
                                       graph_feature.view(bsz, -1, 300))
        # 频域自注意力
        self_att_f = self.mh_attention(fembedding.view(bsz, -1, 300), fembedding.view(bsz, -1, 300), \
                                       fembedding.view(bsz, -1, 300))

        if (sum_probs <= 0.7):
            text_enhanced = self.mh_attention(self_att_f.view((bsz, -1, 300)),  self_att_t.view((bsz, -1, 300)), \
                                          self_att_t.view((bsz, -1, 300))).view(bsz, 300)
        else:
            text_enhanced = self.mh_attention(img_enhanced.view((bsz, -1, 300)), self_att_t.view((bsz, -1, 300)), \
                                          self_att_t.view((bsz, -1, 300))).view(bsz, 300)


        align_text = self.alignfc_t(text_enhanced)
        align_rembedding = self.alignfc_g(self_att_g)
        dist = [align_text, align_rembedding]

        self_att_t = text_enhanced.view((bsz, -1, 300))

        # 按模型图从上往下看
        # 1. 增强文和增强图，增强文和网络 以及他俩的反面
        co_att_ti = self.mh_attention(self_att_t, img_enhanced, img_enhanced).view(bsz,300)
        co_att_tg = self.mh_attention(self_att_t, self_att_g, self_att_g).view(bsz,300)

        co_att_it = self.mh_attention(img_enhanced, self_att_t, self_att_t).view(bsz,300)
        co_att_gt = self.mh_attention(self_att_g, self_att_t, self_att_t).view(bsz, 300)

        # 2. 增强图和网络 以及反面
        co_att_ig = self.mh_attention(img_enhanced, self_att_g, self_att_g).view(bsz, 300)
        co_att_gi = self.mh_attention(self_att_g, img_enhanced, img_enhanced).view(bsz, 300)

        att_feature = torch.cat((co_att_ti, co_att_tg, co_att_it, co_att_gt, co_att_ig, co_att_gi),dim = 1)

        a1 = self.relu(self.dropout(self.fc3(att_feature)))
        a1 = self.relu(self.fc4(a1))
        a1 = self.relu(self.fc1(a1))
        d1 = self.dropout(a1)
        output = self.fc2(d1)

        return output, dist

def load_dataset():
    pre = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_files'

    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))

    config['embedding_weights'] = word_embeddings
    config['node_embedding'] = pickle.load(open(pre + "/node_embedding.pkl", 'rb'))[0]
    print("#nodes: ", adj.shape[0])

    with open(pre + '/new_id_dic.json', 'r') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))

    mid2num = {}
    for file in os.listdir(os.path.dirname(os.getcwd()) + '/dataset/weibo/weibocontentwithimage/original-microblog/'):
        mid2num[file.split('_')[-2]] = file.split('_')[0]
    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]

    config['newid2imgnum'] = newid2num

    return X_train_tid, X_train, y_train, \
        X_dev_tid, X_dev, y_dev, \
        X_test_tid, X_test, y_test, adj

def load_original_adj(adj):
    pre = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_files/'
    path = os.path.join(pre, 'original_adj')

    with open(path, 'r') as f:
        original_adj_dict = json.load(f)
    original_adj = np.zeros(shape=adj.shape)
    for i, v in original_adj_dict.items():
        v = [int(e) for e in v]
        original_adj[int(i), v] = 1
    return original_adj


def train_and_test(model):
    # 以下只是在创建一个保存最佳模型的文件夹
    model_suffix = model.__name__.lower().strip("text")
    res_dir = 'exp_result'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)

    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))


    X_train_tid, X_train, y_train, \
        X_dev_tid, X_dev, y_dev, \
        X_test_tid, X_test, y_test, adj = load_dataset()

    original_adj = load_original_adj(adj)
    original_adj1 = original_adj.dot(original_adj)

    original_adj = sparse.coo_matrix(original_adj)
    original_adj1 = sparse.coo_matrix(original_adj1)

    g0 = dgl.from_scipy(original_adj)
    g1 = dgl.from_scipy(original_adj1)
    g0 = dgl.add_self_loop(g0)
    g1 = dgl.add_self_loop(g1)
    g0 = g0.to('cuda:0')
    g1 = g1.to('cuda:0')
    g2 = [g0, g1]

    nn = model(config, adj, g2)

    nn.fit(X_train_tid, X_train, y_train, X_dev_tid, X_dev, y_dev, g2)

    nn.eval()
    nn.fre_embedding.eval()
    with torch.no_grad():
        for i in range(10):
            print("+++++")
            y_pred = nn.predict(X_test_tid, X_test, g2)
            res = classification_report(y_test, y_pred, target_names=config['target_names'], digits=3, output_dict=True)
            for k, v in res.items():
                print(k, v)
            print("result:{:.4f}".format(res['accuracy']))



config = process_config(config_file.config)
seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

model = CAHIA
train_and_test(model)






