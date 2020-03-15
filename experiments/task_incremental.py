from collections import OrderedDict
from dataclasses import dataclass
from itertools import accumulate
from random import shuffle
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from simple_parsing import choice, field, subparsers
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image

from common.losses import LossInfo
from config import Config
from datasets.dataset import TaskConfig
from datasets.subset import VisionDatasetSubset
from experiments.class_incremental import ClassIncremental
from experiments.experiment import Experiment
from utils.utils import n_consecutive


@dataclass
class TaskIncremental(Experiment):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.

    """
    n_classes_per_task: int = 2      # Number of classes per task.
    # Wether to sort out the classes in the class_incremental setting.
    random_class_ordering: bool = True
    epochs_per_task: int = 1  # Number of epochs on each task's dataset.
    
    # Number of runs to execute in order to create the OML Figure 3.
    n_runs: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.train_full_dataset: VisionDataset = None
        self.valid_full_dataset: VisionDataset = None
        self.train_datasets: List[VisionDatasetSubset] = []
        self.valid_datasets: List[VisionDatasetSubset] = []
        self.valid_cumul_datasets: List[VisionDatasetSubset] = []
        self.task_classes: List[List[int]] = list()

    def run(self):
        train_losses_list: List[List[LossInfo]] = []
        valid_losses_list: List[List[LossInfo]] = []
        # TODO: in the OML figure 3, the class accuracy at the end is ordered
        # in the same way as they were learned.
        final_class_accuracies: List[Tensor] = []

        for i in range(self.n_runs):
            print(f"STARTING RUN {i}")
            train_losses, valid_losses, final_class_accuracy = self.do_one_run()
            train_losses_list.append(train_losses)
            valid_losses_list.append(valid_losses)
            final_class_accuracies.append(final_class_accuracy)
        
        def stack_total_losses(losses: List[List[LossInfo]]) -> Tensor:
            n = len(losses)
            length = len(losses[0])
            result = torch.zeros([n, length], dtype=torch.float)
            for i, run_losses in enumerate(losses):
                for j, epoch_loss in enumerate(run_losses):
                    result[i,j] = epoch_loss.total_loss.item()
            return result

        train_loss = stack_total_losses(train_losses_list)
        valid_loss = stack_total_losses(valid_losses_list)
        final_class_accuracy = torch.stack(final_class_accuracies)
        print("TRAINING LOSSES:")
        print(train_loss)
        print("CUMULATIVE VALID LOSS:")
        print(valid_loss)
        print("FINAL ACCURACIES:")
        print(final_class_accuracy)    

        loss_means = train_loss.mean(dim=0).numpy()
        loss_stds = train_loss.std(dim=0).numpy()
        x = np.arange(loss_means.shape[0])
        print("Loss Means:", loss_means)
        print("Loss STDs:", loss_stds)

        accuracy_means = final_class_accuracy.mean(dim=0).numpy()
        accuracy_std =   final_class_accuracy.std(dim=0).numpy()
        n_classes = len(accuracy_means)
        print("Final Class Accuracy means:", accuracy_means)
        print("Final Class Accuracy stds:", accuracy_std)

        fig: plt.Figure = plt.figure()

        ax1: plt.Axes = fig.add_subplot(1, 2, 1)
        ax1.errorbar(x=x, y=loss_means, yerr=loss_stds)
        ax1.set_title("Continual Classification Loss")
        ax1.set_xlabel("Number of tasks learned")
        ax1.set_ylabel("Classification Loss")

        ax2: plt.Axes = fig.add_subplot(1, 2, 2)
        ax2.bar(x=np.arange(n_classes), height=accuracy_means, yerr=accuracy_std)
        ax2.set_title(f"Final Class Accuracies")
        ax2.set_xlabel("Task ID")

        fig.show()
        fig.savefig(self.plots_dir / "oml_fig.jpg")

    
    def do_one_run(self):
        label_order = self.load()
        datasets = zip(
            self.train_datasets,
            self.valid_datasets,
            self.valid_cumul_datasets
        )
        
        train_losses: List[LossInfo] = []
        valid_losses: List[LossInfo] = []

        for task_index, (train, valid, valid_cumul) in enumerate(datasets):
            print(f"Starting task {task_index}, Classes {self.task_classes[task_index]}")
            kwargs: Dict = {
                "batch_size": self.hparams.batch_size,
                "shuffle": False,
                "num_workers": 1,
                "pin_memory": self.config.use_cuda,
            }
            train_loader = DataLoader(train, **kwargs)
            valid_loader = DataLoader(valid, **kwargs)
            cumul_loader = DataLoader(valid_cumul, **kwargs)

            # Train on the current task:
            self.train_on_task(train_loader, valid_loader)
            
            # Evaluate the performance on training set.
            # We only really do this in order to get a plot like the one in OML.
            train_loss = sum(self.test_iter(f"Task {task_index}", train_loader), LossInfo())
            
            # Evaluate the performance on the cumulative validation set.
            valid_loss = sum(self.test_iter(f"Task {task_index} (cumul)", cumul_loader), LossInfo())

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            validation_metrics = valid_loss.metrics
            class_accuracy = validation_metrics.class_accuracy  
            print(f"AFTER TASK {task_index}: ",
                  f"\tTrain loss: {train_loss.total_loss}, ",
                  f"\tCumulative Validation Loss: {valid_loss.total_loss}, ",
                  f"\tClass Accuracy: {class_accuracy} ", sep="\n")

        # Get the most recent class accuracy metrics.
        return train_losses, valid_losses, class_accuracy[label_order]


    def load(self):
        """ Create the datasets for each task. """
        # download the dataset.
        self.dataset.load(self.config)

        assert self.dataset.train is not None
        assert self.dataset.valid is not None

        # safeguard the entire training dataset.
        self.train_full_dataset = self.dataset.train
        self.valid_full_dataset = self.dataset.valid
        self.train_datasets.clear()
        self.valid_datasets.clear()
        self.task_classes.clear()
        all_labels = list(range(self.dataset.y_shape[0]))
        
        if self.random_class_ordering:
            shuffle(all_labels)
        print("Class Ordering:", all_labels)
        
        for label_group in n_consecutive(all_labels, self.n_classes_per_task):
            train = VisionDatasetSubset(self.train_full_dataset, label_group)
            self.train_datasets.append(train)

            valid = VisionDatasetSubset(self.valid_full_dataset, label_group)
            self.valid_datasets.append(valid)

            self.task_classes.append(list(label_group))

        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
        
        for i, dataset in enumerate(self.train_datasets):
            self.save_images(i, dataset, prefix="train_")
        
        for i, dataset in enumerate(self.valid_datasets):
            self.save_images(i, dataset, prefix="valid_")
        
        for i, dataset in enumerate(self.valid_cumul_datasets):
            self.save_images(i, dataset, prefix="valid_cumul_")
        
        return all_labels
        

    def train_on_task(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        """ Train on a given task.
        
        The given `train_loader` and `valid_loader` DataLoaders are only for the
        current task.
        """
        for epoch in range(self.epochs_per_task):
            # Train for an epoch.
            # NOTE: train_iter logs the train_loss periodically to wandb with the global_step.
            for train_loss in self.train_iter(epoch, train_loader):
                pass

            # Evaluate the performance on validation set.
            valid_epoch_loss = sum(self.test_iter(epoch, valid_loader), LossInfo())

            # log the validation loss for this epoch.
            self.log(valid_epoch_loss, prefix="Valid ")

            total_validation_loss = valid_epoch_loss.total_loss.item()
            validation_metrics = valid_epoch_loss.metrics

       
    def save_images(self, i: int, dataset: VisionDatasetSubset, prefix: str=""):
        n = 64
        samples = dataset.data[:n].view(n, *self.dataset.x_shape).float()
        save_image(samples, self.samples_dir / f"{prefix}task_{i}.png")
        
    def set_task(self, task_index: int) -> Tuple[DataLoader, DataLoader]:
        assert 0 <= task_index < len(self.train_datasets)
        self.dataset.train = self.train_datasets[task_index]
        self.dataset.valid = self.valid_cumul_datasets[task_index]
        ## equivalent to super().load()
        dataloaders = self.dataset.get_dataloaders(self.config, self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders
        return self.train_loader, self.valid_loader

