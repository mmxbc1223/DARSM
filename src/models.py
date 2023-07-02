import torch.nn as nn
from global_config import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
class DARSM(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(DARSM, self).__init__()
        self.fcl = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.fcv = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.fca = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.rnnv = RNNEncoder()
        self.rnna = RNNEncoder()

        self.enc_domain = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        ) 
        self.enc_sentiment = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        ) 

        self.xlgen = nn.Sequential(
            nn.Linear(2*TEXT_DIM, TEXT_DIM),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        )
        self.xvgen = nn.Sequential(
            nn.Linear(2*TEXT_DIM, TEXT_DIM),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        )
        self.xagen = nn.Sequential(
            nn.Linear(2*TEXT_DIM, TEXT_DIM),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        )
        self.recon_l1_loss = nn.L1Loss()
        self.mi = MinMI(TEXT_DIM, TEXT_DIM, TEXT_DIM // 4)
        self.classfier_reg_senti_senti = nn.Sequential(
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.Linear(TEXT_DIM, 1)
        ) 
        self.disc_senti_domain = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM // 2),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(TEXT_DIM // 2, 3)
        ) 
        self.classfier_reg_domain_senti = nn.Sequential(
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.Linear(TEXT_DIM, 1)
        ) 
        self.disc_domain_domain = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM // 2),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(TEXT_DIM // 2, 3)
        )

        self.pooll = Pooler(dim=TEXT_DIM)
        self.poolv = Pooler(dim=TEXT_DIM)
        self.poola = Pooler(dim=TEXT_DIM)

        self.mse = nn.MSELoss()
        self.adv_loss = nn.CrossEntropyLoss()
    def forward(self, text_embedding, visual, acoustic, visual_len, acoustic_len, input_mask_mix=None, training=True, label_ids=None):
        # lengths = (input_mask_mix == 1).sum(dim=1).cpu()
        bsz, seq, dim = text_embedding.shape
        xl, xv, xa = self.fcl(text_embedding), self.fcv(visual), self.fca(acoustic)
        xv, xa = xv.transpose(0, 1), xa.transpose(0, 1)
        xv, xa = self.rnnv(xv, visual_len), self.rnna(xa, acoustic_len)
        xl = xl[:,0,:]
        
        xls, xld = self.enc_sentiment(xl), self.enc_domain(xl)
        xvs, xvd = self.enc_sentiment(xv), self.enc_domain(xv)
        xas, xad = self.enc_sentiment(xa), self.enc_domain(xa)
        # recon loss
        xlg = self.xlgen(torch.cat([xls, xld], dim=-1))
        xvg = self.xvgen(torch.cat([xvs, xvd], dim=-1))
        xag = self.xagen(torch.cat([xas, xad], dim=-1))
        recon_loss_l = self.recon_l1_loss(xl, xlg)
        recon_loss_v = self.recon_l1_loss(xv, xvg)
        recon_loss_a = self.recon_l1_loss(xa, xag)
        recon_loss = recon_loss_l + recon_loss_v + recon_loss_a
        # mi loss
        mi_loss_l = self.mi.learning_loss(xls, xvd)
        mi_loss_v = self.mi.learning_loss(xvs, xvd)
        mi_loss_a = self.mi.learning_loss(xas, xad)
        mi_loss = mi_loss_l + mi_loss_a + mi_loss_v
        # adc loss
        # modality label
        label_l = torch.zeros(bsz, dtype=torch.long).to(DEVICE)
        label_v = torch.ones(bsz, dtype=torch.long).to(DEVICE)
        label_a = label_v.data.new(label_v.size()).fill_(2)
        # text adv + senti
        logits_l_s_s = self.classfier_reg_senti_senti(xls)
        s_s_loss_l = self.mse(logits_l_s_s.view(-1), label_ids.view(-1))
        xlsd_pred = self.disc_senti_domain(GradReverse.grad_reverse(xls, 1)).view(-1, 3)
        s_d_loss_l = self.adv_loss(xlsd_pred, label_l)
    
        logits_l_d_s = self.classfier_reg_domain_senti(GradReverse.grad_reverse(xld, 1))
        d_s_loss_l = self.mse(logits_l_d_s.view(-1), label_ids.view(-1))
        xldd_pred = self.disc_domain_domain(xld).view(-1, 3)
        d_d_loss_l = self.adv_loss(xldd_pred, label_l)

        # visual adv + senti
        logits_v = self.classfier_reg_senti_senti(xvs)
        s_s_loss_v = self.mse(logits_v.view(-1), label_ids.view(-1))
        xvsd_pred = self.disc_senti_domain(GradReverse.grad_reverse(xvs, 1)).view(-1, 3)
        s_d_loss_v = self.adv_loss(xvsd_pred, label_v)

        logits_v_d_s = self.classfier_reg_domain_senti(GradReverse.grad_reverse(xvd, 1))
        d_s_loss_v = self.mse(logits_v_d_s.view(-1), label_ids.view(-1))
        xvdd_pred = self.disc_domain_domain(xvd).view(-1, 3)
        d_d_loss_v = self.adv_loss(xvdd_pred, label_v)

        #acoustic adv + senti
        logits_a = self.classfier_reg_senti_senti(xas)
        s_s_loss_a = self.mse(logits_a.view(-1), label_ids.view(-1))
        xasd_pred = self.disc_senti_domain(GradReverse.grad_reverse(xas, 1)).view(-1, 3)
        s_d_loss_a = self.adv_loss(xasd_pred, label_a)

        logits_a_d_s = self.classfier_reg_domain_senti(GradReverse.grad_reverse(xad, 1))
        d_s_loss_a = self.mse(logits_a_d_s.view(-1), label_ids.view(-1))
        xadd_pred = self.disc_domain_domain(xad).view(-1, 3)
        d_d_loss_a = self.adv_loss(xadd_pred, label_a)

        s_s_loss = s_s_loss_l + s_s_loss_v + s_s_loss_a
        d_d_loss = d_d_loss_l + d_d_loss_v + d_d_loss_a
        s_d_loss = s_d_loss_l + s_d_loss_v + s_d_loss_a
        d_s_loss = d_s_loss_l + d_s_loss_v + d_s_loss_a 
        grl_loss = s_s_loss + d_d_loss + s_d_loss + d_s_loss
        xl1, xv1, xa1, xl2, xv2, xa2 = xls, xvs, xas, xld, xvd, xad

        xls, xvs, xas = self.pooll(xls), self.poolv(xvs), self.poola(xas)
        return xls, xvs, xas, mi_loss, recon_loss, grl_loss, xl1, xv1, xa1, xl2, xv2, xa2, logits_l_s_s, logits_v, logits_a

class RNNEncoder(nn.Module):
    def __init__(self, in_size=768, hidden_size=768, out_size=768, num_layers=1, dropout=0.2, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.to(torch.int64)
        bs = x.size(0)

        packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        
        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0],final_states[0][1]),dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
    
class Pooler(nn.Module):
    def __init__(self, dim):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(dim, dim)
        self.activation = nn.Tanh()
    def forward(self, x):
        pooled_output = self.dense(x)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class MinMI(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MinMI, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          
        y_samples_1 = y_samples.unsqueeze(0)    

        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)