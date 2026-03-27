import copy
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset

from model.lightning_wrapper import LightningWrapper
from model.prithvi import PrithviModel

from data.ravaen import RavaenDataset
from data.sttorm import SttormDataset
from data.augmentations import augmentation_from_dict
import yaml

def load_data(cfg):
    train_transforms = augmentation_from_dict(cfg['augmentations']['train'])
    test_transforms = augmentation_from_dict(cfg['augmentations']['test'])

    assert isinstance(cfg['train'], list), "cfg['train'] must be a list of datasets"
    train_sets = []
    val_sets = []
    for ds_cfg in cfg['train']:
        CLS = eval(ds_cfg['type'])
        ds = CLS(ds_cfg['path'], csv_path=ds_cfg['csv'], transforms=train_transforms, **cfg["args"], **cfg["cutmix"])
        print("Train:", ds_cfg['type'], ds.data['event'].unique(), len(ds))
        ds_train, ds_val = ds.random_split(test_size=0.2, seed=cfg['seed'])
        ds_val.transforms = test_transforms  
        train_sets.append(ds_train)
        val_sets.append(ds_val)

    train_set = ConcatDataset(train_sets)
    val_set = ConcatDataset(val_sets)
    
    CLS = eval(cfg['test']['type'])
    test_set = CLS(cfg['test']['path'], csv_path=cfg['test']['csv'], transforms=test_transforms, **cfg["args"])
    print("Test:", cfg['test']['type'], test_set.data['event'].unique(), len(test_set))
    
    return train_set, val_set, test_set

def get_model_from_cfg(cfg):
    unet_cfg = cfg['model']['unet']
    unet = PrithviModel(n_classes=cfg['model']['args']['n_classes'], **unet_cfg['args'])
    module = LightningWrapper(unet, max_epochs=cfg['epochs'], **cfg['model']['args'])
    return module

def get_options_for_experiment(cfg):
    name = cfg['name']
    version = cfg['model']['args']['version']
    
    if name == "Fuse Stage":
        # STAGES = [11, 7, 5, 3]
        STAGES = [7, 5, 3]
        for s in STAGES:
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy['model']['unet']['args']['fuse_stage'] = s
            cfg_copy['name'] = f"{name}-{s}"
            yield cfg_copy
    elif name == "Dim":
        print("--------------------------")
        print("Generating Dim Reduction configs")
        print("--------------------------")
        # STAGES = [0,24,48,72,96,120,144,168,192]
        STAGES = [12, 8, 4, 1]
        for s in STAGES:
            cfg_copy = copy.deepcopy(cfg)
            assert 'embed_dim' in cfg_copy['model']['unet']['args'], "embed_dim not in config"
            cfg_copy['model']['unet']['args']['embed_dim'] = s
            cfg_copy['name'] = f"{name}-{s}"
            yield cfg_copy
    elif name == "Spatial":
        print("--------------------------")
        print("Generating Spatial configs")
        print("--------------------------")
        SPATIALS = [16, 8, 4, 2]
        for s in SPATIALS:
            cfg_copy = copy.deepcopy(cfg)
            assert 'embed_tokens' in cfg_copy['model']['unet']['args'], "spatial_size not in config"
            cfg_copy['model']['unet']['args']['embed_tokens'] = s**2
            cfg_copy['name'] = f"{name}-{s}"
            yield cfg_copy
    else:
        print("------------------------------------------------------------------")
        print(f" Using base config for experiment: {name} and version: {version}")
        print("------------------------------------------------------------------")
        yield cfg

if __name__ == "__main__":
    import argparse

    WANDB_API_KEY  = "<YOUR-API-KEY>"
    CHECKPOINT_DIR = "./"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    
    if cfg['wandb']:
        with open(WANDB_API_KEY, 'r') as f:
            wandb.login(key=f.read().strip())

    ds_train, ds_val, ds_test = load_data(cfg['data'])

    cfg_copy = copy.deepcopy(cfg)
    for cfg in get_options_for_experiment(cfg_copy):
        for idx in range(cfg['n_runs']):

            dl_train = DataLoader(ds_train, cfg['batch_size'], shuffle=True, num_workers=cfg['n_workers'])
            dl_val = DataLoader(ds_val, cfg['batch_size'], shuffle=False, num_workers=cfg['n_workers'])
            dl_test = DataLoader(ds_test, cfg['batch_size'], shuffle=False, num_workers=cfg['n_workers'])

            module = get_model_from_cfg(cfg)

            logger = None
            if cfg['wandb']:
                logger = WandbLogger(project=cfg['project'], name=f'{cfg["name"]}', log_model=False)
                logger.log_hyperparams(cfg)
                logger.log_hyperparams({"NET": cfg['model']['unet']})

                
            callbacks =[
                checkpoint_cb := ModelCheckpoint(**cfg['checkpoint'],dirpath=CHECKPOINT_DIR, filename=f"{cfg['name']}#{idx+1}" + "-{epoch:02d}-{val_loss:.2f}"),
            ]
            if 'early_stopping' in cfg:
                callbacks.append(EarlyStopping(**cfg['early_stopping']))

            trainer = Trainer(accelerator='gpu', 
                            devices=[0],
                            max_epochs=cfg['epochs'],
                            logger=logger,
                            callbacks=callbacks,
                            inference_mode=False,
                            log_every_n_steps=20,
                            )

            trainer.fit(module, dl_train, dl_val)
            module.mode = 'last'
            trainer.test(module, dl_test)
            module.mode = 'train'
            trainer.test(module, dl_test, checkpoint_cb.best_model_path)

            if cfg['wandb']:
                wandb.finish()

