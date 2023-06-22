import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import model
import dataset
from src.utils import set_logger, get_pck_with_sigma, get_pred_coordinates
from src.loss import sum_mse_loss
from easydict import EasyDict as edict
import numpy as np
import matplotlib.pyplot as plt

currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)
# ***********************  Parameter  ***********************

args = edict({
    "config_file": 'data_sample/Panoptic_base.json',
    "GPU": 0
})
configs = json.load(open(args.config_file)) # type: ignore

target_sigma_list = [0.05, 0.1, 0.15, 0.2, 0.25]
select_sigma = 0.15

model_name = 'EXP_' + configs["name"]
torch.cuda.empty_cache()
save_dir = os.path.join(model_name, 'checkpoint/')
test_pck_dir = os.path.join(model_name, 'test/')

if os.path.exists(model_name):
    print("the log directory exists!")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(test_pck_dir, exist_ok=True)

# training parameters ****************************
data_root = configs["data_root"]
learning_rate = configs["learning_rate"]
batch_size = configs["batch_size"]
epochs = configs["epochs"]

# data parameters ****************************

cuda = False


logger = set_logger(os.path.join(model_name, 'train.log'))
logger.info("************** Experiment Name: {} **************".format(model_name))

# ******************** build model ********************
logger.info("Create Model ...")

model = model.light_Model(configs)
if cuda:
    model = model.cuda()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.numel())
    return {'Total': total_num, 'Trainable': trainable_num}
print('the number of params:', get_parameter_number(model))

# ******************** data preparation  ********************
my_dataset = getattr(dataset, configs["dataset"])
train_data = my_dataset(data_root=data_root, mode='train')
valid_data = my_dataset(data_root=data_root, mode='valid')
test_data = my_dataset(data_root=data_root, mode='test')
logger.info('Total images in training data is {}'.format(len(train_data)))
logger.info('Total images in validation data is {}'.format(len(valid_data)))
logger.info('Total images in testing data is {}'.format(len(test_data)))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)


# ********************  ********************
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

def plot_loss(train_losses, valid_losses):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.xticks(epochs)
    plt.show()

def plot_pck(pck_values):
    epochs = np.arange(1, len(pck_values) + 1)
    plt.figure()
    plt.plot(epochs, pck_values)
    plt.xlabel('Epoch')
    plt.ylabel('PCK')
    plt.title('PCK Evolution')
    plt.xticks(epochs)
    plt.show()

train_losses = []
valid_losses = []
pck_values = []

def train(model):
    logger.info('\nStart training ===========================================>')
    best_epo = -1
    max_pck = -1
    logger.info('Initial learning Rate: {}'.format(learning_rate))

    for epoch in range(1, epochs + 1):
        logger.info('Epoch[{}/{}] ==============>'.format(epoch, epochs))
        train_label_loss = []
        model.train()
        for step, data in enumerate(train_loader, 0):
            # *************** target prepare ***************
            img, label_terget, img_name, w, h = data
            if cuda:
                img = img.cuda(non_blocking=True)
                label_terget = label_terget.cuda(non_blocking=True)
            optimizer.zero_grad()
            label_pred = model(img)

            # *************** calculate loss ***************
            label_loss = sum_mse_loss(label_pred.float(), label_terget.float())  # keypoint confidence loss
            label_loss.backward()
            optimizer.step()

            train_label_loss.append(label_loss.item())

            label_loss.detach()

            if step % 10 == 0:
                logger.info('TRAIN STEP: {}  LOSS {}'.format(step, label_loss.item()))

        # *************** eval model after one epoch ***************
        eval_loss, cur_pck = eval(model, epoch, mode='valid') # type: ignore
        train_losses.append(sum(train_label_loss) / len(train_label_loss))
        valid_losses.append(eval_loss)
        pck_values.append(cur_pck)
        logger.info('EPOCH {} VALID PCK  {}'.format(epoch, cur_pck))
        logger.info('EPOCH {} TRAIN_LOSS {}'.format(epoch, sum(train_label_loss) / len(train_label_loss)))
        logger.info('EPOCH {} VALID_LOSS {}'.format(epoch, eval_loss))

        # *************** save current model and best model ***************
        if cur_pck > max_pck:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            torch.save(optimizer.state_dict(),  os.path.join(save_dir,'best_optimizador.pth'))
            best_epo = epoch
            max_pck = cur_pck
        logger.info('Current Best EPOCH is : {}, PCK is : {}\n**************\n'.format(best_epo, max_pck))

        # save current model
        torch.save(model.state_dict(), os.path.join(save_dir, 'epoch_' + str(epoch) + '_' + str(cur_pck) + '.pth'))
        torch.save(optimizer.state_dict(),  os.path.join(save_dir,'epoch_' + str(epoch) + str(cur_pck) + '_''optimizador.pth'))
        # scheduler
        scheduler.step(cur_pck)

    logger.info('Train Done! ')
    logger.info('Best epoch is {}'.format(best_epo))
    logger.info('Best Valid PCK is {}'.format(max_pck))


def eval(model, mode='valid'):
    if mode == 'valid':
        loader = valid_loader
        gt_labels = valid_data.all_labels
    else:
        loader = test_loader
        gt_labels = test_data.all_labels

    with torch.no_grad():
        all_pred_labels = {}  # save predict results
        eval_loss = []
        model.eval()
        for step, (img, label_terget, img_name, w, h) in enumerate(loader):
            if cuda:
                img = img.cuda(non_blocking=True)
            cm_pred = model(img)

            all_pred_labels = get_pred_coordinates(cm_pred.cpu(), img_name, w, h, all_pred_labels)
            loss_final = sum_mse_loss(cm_pred.cpu(), label_terget)
            if step % 10 == 0:
                logger.info('EVAL STEP: {}  LOSS {}'.format(step, loss_final.item()))
            eval_loss.append(loss_final)

        # ************* calculate PCKs  ************
        pck_dict = get_pck_with_sigma(all_pred_labels, gt_labels, target_sigma_list)

        select_pck = pck_dict[select_sigma]
        eval_loss = sum(eval_loss) / len(eval_loss)
    return eval_loss, select_pck


train(model)

logger.info('\nTESTING ============================>')
logger.info('Load Trained model !!!')
state_dict = torch.load(os.path.join(save_dir, 'best_model.pth'))
model.load_state_dict(state_dict)
eval(model, mode='test')

logger.info('Done!')

plot_loss(train_losses, valid_losses)
plot_pck(pck_values)