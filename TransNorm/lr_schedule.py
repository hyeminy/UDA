

def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_deacy=0.0005):

    """
    lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    =
    inv_lr_scheduler(optimizer_config["lr_type"])

    inv_lr_scheduler({"lr": args.lr,
                      'gamma': 0.001, # 이거 10이여야 하지 않나?
                      'power': 0.75
                      }
    """

    lr = lr * (1+gamma*iter_num)**(-power)

    i = 0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_deacy * param_group['decay_mult']
        i += 1

    return optimizer


schedule_dict = {"inv" : inv_lr_scheduler}