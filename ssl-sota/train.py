from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
import os 

from cfg import get_cfg
from datasets import get_ds
from methods import get_method


def get_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epoch if cfg.T0 is None else cfg.T0,
            T_mult=cfg.Tmult,
            eta_min=cfg.eta_min,
        )
    elif cfg.lr_step == "step":
        m = [cfg.epoch - a for a in cfg.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.drop_gamma)
    else:
        return None


if __name__ == "__main__":
    cfg = get_cfg()
    wandb.init(project=f"ssl-sota-{cfg.method}-{cfg.dataset}", config=cfg, dir='/mnt/store/wbandar1/projects/ssl-aug-artifacts/wandb_logs/')
    run_id = wandb.run.id
    # if not os.path.exists('../results'):
    #     os.mkdir('../results')
    run_id_dir = os.path.join('/mnt/store/wbandar1/projects/ssl-aug-artifacts/', run_id)
    if not os.path.exists(run_id_dir):
        print('Creating directory {}'.format(run_id_dir))
        os.mkdir(run_id_dir)

    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers)
    model = get_method(cfg.method)(cfg)
    model.cuda().train()
    if cfg.fname is not None:
        model.load_state_dict(torch.load(cfg.fname))

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)
    scheduler = get_scheduler(optimizer, cfg)

    eval_every = cfg.eval_every
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True

    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        iters = len(ds.train)
        for n_iter, (samples, _) in enumerate(tqdm(ds.train, position=1)):
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1

            optimizer.zero_grad()
            loss = model(samples)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            model.step(ep / cfg.epoch)
            if cfg.lr_step == "cos" and lr_warmup >= 500:
                scheduler.step(ep + n_iter / iters)

        if cfg.lr_step == "step":
            scheduler.step()

        if len(cfg.drop) and ep == (cfg.epoch - cfg.drop[0]):
            eval_every = cfg.eval_every_drop

        if (ep + 1) % eval_every == 0:
            # acc_knn, acc = model.get_acc(ds.clf, ds.test)
            # wandb.log({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}, commit=False)
            acc_knn = model.get_acc_knn(ds.clf, ds.test)
            wandb.log({"acc_knn": acc_knn}, commit=False)

        if (ep + 1) % 100 == 0:
            fname = f"/mnt/store/wbandar1/projects/ssl-aug-artifacts/{run_id}/{cfg.method}_{cfg.dataset}_{ep}.pt"
            torch.save(model.state_dict(), fname)
        wandb.log({"loss": np.mean(loss_ep), "ep": ep})
    
    acc_knn, acc = model.get_acc(ds.clf, ds.test)
    print('Final linear-acc: {}, knn-acc'.format(acc, acc_knn))
    wandb.log({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}, commit=False)
    wandb.finish()