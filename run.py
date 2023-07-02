from PIL import Image
import torch
import wandb
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from global_config import *
from src.models import *
from transformers.models.bert.tokenization_bert import BertTokenizer
import argparse
from utils.utils import *
import pickle
from dataset.dataset import *
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--cuda_no", type=str, default=os.environ["CUDA_VISIBLE_DEVICES"])
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default=DATASETS)
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=EPOCHS)
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")
parser.add_argument("--best_acc", type=float, default=0.1)
args = parser.parse_args()

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def prepare_training(train_dataloader):
    model = DARSMModel()
    model.to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCHS)
    return model, optimizer, scheduler



def train_epoch(model, train_dataloader, optimizer, scheduler='full'):
    step = 0
    tr_loss = 0
    train_cl_loss = 0
    train_task_loss = 0
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        sentence, input_ids, attention_mask, visual, visual_len, label_id, acoustic, acoustic_len, segment = batch
        input_ids, attention_mask, visual, acoustic, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), acoustic.to(DEVICE), label_id.to(DEVICE)
        step += 1
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, visual, visual_len, acoustic, acoustic_len)
        logits = outputs
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_id.view(-1))
        total_loss = loss
        total_loss.backward()
        tr_loss += total_loss.item()
        #convert_models_to_fp32(model)
        optimizer.step()
        scheduler.step()
        #clip.model.convert_weights(model)
    tr_loss /= step
    train_cl_loss /= step
    train_task_loss /= step
    return tr_loss

def eval_epoch(model, dev_dataloader, optimizer='full'):
    step = 0
    dev_loss = 0
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, visual, visual_len, label_id, segment = batch
            input_ids, attention_mask, visual, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), label_id.to(DEVICE)
            step += 1
            outputs = model(input_ids, attention_mask, visual, visual_len, acoustic, acoustic_len)
            logits = outputs
            
            label_ids = label_id.detach().cpu().numpy()
            label_ids = np.squeeze(label_ids).tolist()

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_id.view(-1))
            total_loss = loss
            dev_loss += total_loss.item()

            logits = logits.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            preds.extend(logits)
            labels.extend(label_ids)
        preds = np.array(preds)
        labels = np.array(labels)
        
        dev_loss /= step
    return dev_loss, preds, labels

def test_epoch(model: nn.Module, test_dataloader: DataLoader, epoch=-1):
    model.eval()
    preds = []
    labels = []
    test_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, visual, visual_len, acoustic, acoustic_len, label_id, segment = batch
            input_ids, attention_mask, visual, acoustic, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), acoustic.to(DEVICE), label_id.to(DEVICE)
            outputs = model(input_ids, attention_mask, visual, visual_len, acoustic, acoustic_len)
            label_ids = label_id.detach().cpu().numpy()
            label_ids = np.squeeze(label_ids).tolist()
            logits = outputs
            
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_id.view(-1))
            total_loss = loss
            test_loss += total_loss.item()

            logits = logits.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            preds.extend(logits)
            labels.extend(label_ids)
        preds = np.array(preds)
        labels = np.array(labels)
    return preds, labels, test_loss





def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False, epoch=-1, test=False):
    # test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    # test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    # test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    # test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
    # acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    # acc5 = multiclass_acc(test_preds_a5, test_truth_a5)

    preds, y_test, test_loss = test_epoch(model, test_dataloader, epoch=epoch)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])
    preds = preds[non_zeros]
    y_test = y_test[non_zeros]
    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    preds = preds >= 0
    y_test = y_test >= 0
    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)
    return acc, mae, corr, f_score, test_loss 




def get_dataset():
    tokenizer = BERTTokenizer.from_pretrained(PRETRAIN_PATH)
    data_path = os.path.join(PATH, PREFIX, DATASETS)
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    train_dataset = data['train']
    dev_dataset = data['dev']
    test_dataset = data['test']
    train_data = MultimodalDataset(train_dataset, tokenizer)
    dev_data = MultimodalDataset(dev_dataset, tokenizer)
    test_data = MultimodalDataset(test_dataset, tokenizer)
    return train_data, dev_data, test_data




def get_dataloader(train_dataset, dev_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    num_train_optimization_steps = 0
    return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps



def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
):
    train_losses, valid_losses, test_losses = [], [], [] 
    valid_accuracies, valid_f_scores, valid_maes, valid_ccs = [], [], [], []
    test_accuracies, test_f_scores, test_maes, test_ccs = [], [], [], []

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_acc, valid_mae, valid_corr, valid_f_score, valid_loss = test_score_model(model, validation_dataloader)
        test_acc, test_mae, test_corr, test_f_score, test_loss = test_score_model(model, test_dataloader)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)
        valid_accuracies.append(valid_acc)
        valid_maes.append(valid_mae)
        valid_ccs.append(valid_corr)
        valid_f_scores.append(valid_f_score)
        test_accuracies.append(test_acc)
        test_maes.append(test_mae)
        test_ccs.append(test_corr)
        test_f_scores.append(test_f_score)
        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_loss": test_loss,
                    "best_valid_loss": min(valid_losses),
                    "best_test_loss": min(test_losses),
                    "best_test_acc": max(test_accuracies),
                    "best_test_f_score": max(test_f_scores),
                    "best_test_corr": max(test_ccs),
                    "best_test_mae": min(test_maes),
                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score": test_f_score,
                    "best_valid_acc": max(valid_accuracies),
                    "best_valid_f_score": max(valid_f_scores),
                    "best_valid_corr": max(valid_ccs),
                    "best_valid_mae": min(valid_maes),
                    "valid_acc": valid_acc,
                    "valid_mae": valid_mae,
                    "valid_corr": valid_corr,
                    "valid_f_score": valid_f_score,
                }
            )
        )


def main():
    wandb.init(project="DARSM-ProJ")
    wandb.config.update(args)
    set_random_seed(args.seed)
    # full setting
    train_dataset, dev_dataset, test_dataset = get_dataset()
    train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps = get_dataloader(train_dataset, dev_dataset, test_dataset)
    model, optimizer, scheduler = prepare_training(train_dataloader)    
    train(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=dev_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer, 
        scheduler=scheduler,
        )

if __name__ == "__main__":
    main()




    