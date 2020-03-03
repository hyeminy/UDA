from mdd_options import config

from torch.autograd import Variable
import torch
from  model.MDD import MDD
from preprocess.data_provider import load_images
import tqdm
import os.path

def train(config):

    train_source_loader = load_images(config['source_path'],
                                      batch_size=config['batch_size'],
                                      resize_size = config['resize_size'],
                                      is_cen = config['is_cen'],
                                      )
    train_target_loader = load_images(config['target_path'],
                                      batch_size=config['batch_size'],
                                      resize_size=config['resize_size'],
                                      is_cen=config['is_cen'],)
    test_target_loader = load_images(config['target_path'],
                                     batch_size=config['batch_size'],
                                     resize_size=config['resize_size'],
                                     is_cen=config['is_cen'],
                                     is_train=False)

    model_instance = MDD(base_net=config['base_net'],
                         width = config['width'],
                         use_gpu = True,
                         class_num = config['class_num'],
                         srcweight = config['srcweight'])

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert config['optim']['type'] == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **config['optim']['params'])

    assert config['lr_scheduler']['type'] == 'inv', 'Scheduler type not supported!'

    lr_scheduler = INVScheduler(gamma = config['lr_scheduler']['gamma'],
                                decay_rate = config['lr_scheduler']['decay_rate'],
                                init_lr = config['init_lr'])

    model_instance.set_train(True)

    print("start train...")
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=config['max_iter'])

    while True:
        for (datas, datat) in tqdm.tqdm(
                zip(train_source_loader, train_target_loader),
                total=min(len(train_source_loader),
                          len(train_target_loader)),
                desc='Train epoch = {}'.format(epoch),
                ncols=80, leave=False):

            inputs_source, labels_source = datas
            inputs_target, labels_target = datat

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num / 5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(
                    inputs_target).cuda(), Variable(labels_source).cuda()
            else:
                inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(
                    inputs_target), Variable(labels_source)

            train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer)

            # val
            if iter_num % config['eval_iter'] == 0 and iter_num !=0:
                eval_result = evaluate(model_instance,
                                       test_target_loader,)
                print(eval_result)
                acc = round(eval_result['accuracy'], 2)
                print(acc)

                torch.save(model_instance.c_net.state_dict(), os.path.join(config["output_path"], str(acc*100)+"_model.pth.tar"))
            iter_num += 1
            total_progress_bar.update(1)

        epoch += 1
        if iter_num >= config['max_iter']:
            break
    print('finish train')









def train_batch(model_instance,inputs_source, labels_source, inputs_target, optimizer):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source)
    total_loss.backward()
    optimizer.step()

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer


def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]

        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        probabilities = model_instance.predict(inputs)
        probabilities = probabilities.data.float()

        labels = labels.data.float()

        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities),0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])

    model_instance.set_train(ori_train_state)
    return {'accuracy' : accuracy}



if __name__ == '__main__':
    train(config)