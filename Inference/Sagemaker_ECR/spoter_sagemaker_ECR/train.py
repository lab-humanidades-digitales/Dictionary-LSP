import argparse
from utils import parse_arguments_automated, set_seed, configure_model, get_dataset, get_datasets_by_dsname
from spoter.training_spoter import TrainingSpoter
from spoter.experimenter import ExperimenterSpoter
import sys
import os
import wandb

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "spoter"
ENTITY = "joeNatan30"

def is_there_arg(args, master_arg):
    if(master_arg in args):
        return True
    else:
        return False

def parse_argument(args, master_arg):
    try:
        if(master_arg in args and args.index(master_arg)+1<len(args)):
            arg  = args[args.index(master_arg)+1]
            return arg
        else:
            return None
    except:
        print("Something went wrong when loading the parameters, Kindly check input carefully!!!")


def train(config_file, use_wandb, exp_name, exp_notes, experimentation, num_logs):
    set_seed(32)
    config = configure_model(config_file, use_wandb)

    if experimentation:
        print("Training some experiments to numerical experimentation ...")
        dict_train_loader, dict_val_loader, dict_eval_loader, dict_dict_labels_dataset, dict_inv_dict_labels_dataset = get_datasets_by_dsname(config, use_wandb)
        exp_spoter = ExperimenterSpoter(
            config, 
            use_wandb, 
            num_logs,
            dict_train_loader,
            dict_val_loader,
            dict_eval_loader,
            dict_dict_labels_dataset,
            dict_inv_dict_labels_dataset
        )
        exp_spoter.train_experiments(entity=ENTITY,
                                    project_wandb=PROJECT_WANDB,
                                    exp_name=exp_name,
                                    exp_notes=exp_notes
        )


    else:
        print("Normal training")
        train_loader, val_loader, eval_loader, dict_labels_dataset, inv_dict_labels_dataset = get_dataset(config, use_wandb)
    
        if use_wandb:
            wandb.init(project=PROJECT_WANDB, entity=ENTITY, config=config, name=exp_name, notes=exp_notes)
            config = wandb.config
            wandb.watch_called = False
            path_save_weights = os.path.join(config.save_weights_path, wandb.run.id + "_" + config.weights_trained)
        else:
            path_save_weights = os.path.join(config.save_weights_path, config.weights_trained)
        try:
            os.mkdir(path_save_weights)
        except OSError:
            pass

        spoter_trainer = TrainingSpoter(config=config, use_wandb=use_wandb,
                                        path_save_weights=path_save_weights
                                        )
        print("Starting training ...")
        spoter_trainer.train(train_loader=train_loader,
                            val_loader=val_loader,
                            eval_loader=eval_loader,
                            dict_labels_dataset=dict_labels_dataset,
                            inv_dict_labels_dataset=inv_dict_labels_dataset
                            )
    

if __name__ == '__main__':
    use_sweep = is_there_arg(sys.argv, '--use_sweep')
    print("Starting training ...")
    if not use_sweep:
        args = parse_arguments_automated()
        use_wandb = args.wandb
        exp_name = args.exp_name
        exp_notes = args.exp_notes
        experimentation = args.experimentation
        num_logs = args.num_logs
    else:
        use_wandb = True
        exp_name = None
        exp_notes = None
        experimentation = parse_argument(sys.argv, '--experimentation')
        num_logs = parse_argument(sys.argv, '--num_logs')

    train(CONFIG_FILENAME, use_wandb=use_wandb, 
        exp_name=exp_name, exp_notes=exp_notes,
        experimentation=experimentation, num_logs=num_logs
        )