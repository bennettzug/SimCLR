import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import accuracy, save_checkpoint, save_config_file

torch.manual_seed(0)


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"].to(self.args.device)
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, "training.log"), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # create a directory for embeddings
        self.embeddings_dir = os.path.join(self.writer.log_dir, "embeddings")
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def extract_embeddings(self, data_loader):
        self.model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Extracting embeddings"):
                img = images[0].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(img)
                    features = F.normalize(features, dim=1)

                embeddings.append(features.cpu().numpy())
                labels.append(targets.numpy())
        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)

        return embeddings, labels

    def extract_labelled_embeddings(self, data_loader):
        self.model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Extracting labelled embeddings"):
                img = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(img)
                    features = F.normalize(features, dim=1)

                embeddings.append(features.cpu().numpy())
                labels.append(targets.numpy())
        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)

        return embeddings, labels

    def train(self, train_loader, labeled_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            self.model.train()

            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar("loss", loss, global_step=n_iter)
                    self.writer.add_scalar("acc/top1", top1[0], global_step=n_iter)
                    self.writer.add_scalar("acc/top5", top5[0], global_step=n_iter)
                    self.writer.add_scalar("learning_rate", self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            if (
                (epoch_counter + 1) % self.args.save_embeddings_every_n_epochs == 0
                or epoch_counter == 0
                or epoch_counter == self.args.epochs - 1
            ):
                logging.info(f"Extracting and saving embeddings for epoch {epoch_counter + 1}")

                # unlabeled
                embeddings, labels = self.extract_embeddings(train_loader)
                embedding_path = os.path.join(self.embeddings_dir, f"embeddings_epoch_{epoch_counter + 1}.npz")
                np.savez(embedding_path, embeddings=embeddings, labels=labels)
                logging.info(f"Unlabeled embeddings saved to {embedding_path}")

                # labeled
                labeled_embeddings, labeled_labels = self.extract_labelled_embeddings(labeled_loader)
                labeled_embedding_path = os.path.join(
                    self.embeddings_dir, f"labeled_embeddings_epoch_{epoch_counter + 1}.npz"
                )
                np.savez(labeled_embedding_path, embeddings=labeled_embeddings, labels=labeled_labels)
                logging.info(f"Labeled embeddings saved to {labeled_embedding_path}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = "checkpoint_{:04d}.pth.tar".format(self.args.epochs)
        save_checkpoint(
            {
                "epoch": self.args.epochs,
                "arch": self.args.arch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            is_best=False,
            filename=os.path.join(self.writer.log_dir, checkpoint_name),
        )
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
