#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

logger = logging.get_logger("visual_prompt")

import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class Trainer:
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """

    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model, save_dir=cfg.OUTPUT_DIR, save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [
                key
                for key in self.checkpointer.checkpointables
                if key not in ["head.last_layer.bias", "head.last_layer.weight"]
            ]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape
                    )
                )

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights, self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(outputs, targets, self.cls_weights)

            if loss == float("inf"):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter("Loss", ":.4e")
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE
        )
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        test_accuracy_ls = []
        loss_ls = []

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(
                        seconds=int(
                            seconds_per_batch * (total_data - idx - 1)
                            + seconds_per_batch * total_data * (total_epoch - epoch - 1)
                        )
                    )
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1, total_data, train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg
                )
                + "average train loss: {:.4f}".format(losses.avg)
            )
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training

            loss_ls.append(losses.avg)

            self.evaluator.update_iteration(epoch)
            test_acc = self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            if test_loader is not None:
                test_acc = self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1
                )

            test_accuracy_ls.append(test_acc)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            # t_name = "test_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][
                    t_name
                ]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(f"Best epoch {best_epoch}: best metric: {best_metric:.3f}")
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

        # save the last checkpoints
        if self.cfg.MODEL.SAVE_CKPT:
            Checkpointer(
                self.model, save_dir=self.cfg.OUTPUT_DIR, save_to_disk=True
            ).save("last_model")

        return test_accuracy_ls[-1]

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if (
                self.cfg.MODEL.TYPE == "vit"
                and "prompt" in self.cfg.MODEL.TRANSFER_TYPE
            ):
                prompt_embds = (
                    self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                )
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = (
                        self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    )
                    out["deep_prompt"] = deep_embds
                torch.save(
                    out, os.path.join(self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth")
                )

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1, total, losses.val, batch_time.val, data_time.val
                    )
                    + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg
            )
            + "average loss: {:.4f}".format(losses.avg)
        )
        if self.model.side is not None:
            logger.info("--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        acc = self.evaluator.classify(
            joint_logits,
            total_targets,
            test_name,
            self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(f"Saved logits and targets for {test_name} at {out_path}")

        return acc

    @torch.no_grad()
    def eval_classifier_GENERAL(
        self, model, train_loader, data_loader, prefix, integrated_method
    ):

        def reshape_transform(tensor, height=14, width=14):
            # print('In reshape_transform', tensor.shape) # [batch size, 197 ,768]
            # print('In reshape_transform', tensor.shape) In reshape_transform torch.Size([64, 297, 768])
            # tensor = tensor[:, 201:, :] # prompt tuning will return value not equal to 197 (with prompt length), reflect in error
            # tensor = tensor[:, 11:, :]
            # result = tensor[:, :, :].reshape(tensor.size(0),
            #                                 height, width, tensor.size(2))
            tensor = tensor[:, 1 + self.cfg.MODEL.PROMPT.NUM_TOKENS :, :]
            result = tensor[:, :, :].reshape(
                tensor.size(0), height, width, tensor.size(2)
            )

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        Checkpointer(model).load(self.cfg.OUTPUT_DIR + "/last_model.pth")

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE
        )

        if integrated_method == "pytorch_gradcam":  # default for visualization gradcam
            method = GradCAM(
                model=model,
                target_layers=[
                    model.enc.transformer.encoder.layer[11].attention_norm
                ],  # [model.enc.transformer.encoder.encoder_norm],
                use_cuda=True,
                reshape_transform=reshape_transform,
            )
        else:
            ValueError(
                f"Unsupported cfg.ATTRIBUTION_INTEGRATED_METHOD in trainer.py: {integrated_method}"
            )

        # wrapper = ParameterWrapper(model.enc.transformer.prompt_embeddings)
        # ig_prompt_embeddings = LayerIntegratedGradients(model, wrapper)

        """evaluate classifier"""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        # grad_prompt = []
        grad_prompt_norm = []
        # grad_embeddings = []
        grad_embeddings_norm = []

        id_iter = itertools.count()

        number_images = 0

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            # Set max 2000 images for visualization
            number_images += X.shape[0]
            if number_images > 500:
                break

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))

            loss, outputs, attribution_patches = self.forward_one_batch_IgGeneral(
                method, X, targets, True, integrated_method
            )  # False

            # save logits
            if self.cfg.SAVE_LOGITS:
                targets_numpy = targets.cpu().numpy()
                targets_numpy.astype(int)
                file_path = f"./{self.cfg.OUTPUT_DIR}/{self.cfg.MODEL.TRANSFER_TYPE}/{self.cfg.ATTRIBUTION_INTEGRATED_METHOD}/t_l_save/Targets_{self.cfg.MODEL.TRANSFER_TYPE}.txt"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "ab") as f:
                    np.savetxt(f, targets_numpy.reshape(1, -1), fmt="%.6f")

                outputs_numpy = outputs.cpu().numpy()
                file_path = f"./{self.cfg.OUTPUT_DIR}/{self.cfg.MODEL.TRANSFER_TYPE}/{self.cfg.ATTRIBUTION_INTEGRATED_METHOD}/t_l_save/Logits_{self.cfg.MODEL.TRANSFER_TYPE}.txt"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "ab") as f:
                    np.savetxt(f, outputs_numpy.reshape(1, -1), fmt="%.6f")

            if attribution_patches is not None:

                default_cmap = LinearSegmentedColormap.from_list(
                    "custom blue",
                    [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")],
                    N=256,
                )
                if integrated_method == "pytorch_gradcam":
                    if not os.path.exists(
                        f"./{self.cfg.OUTPUT_DIR}/{self.cfg.MODEL.TRANSFER_TYPE}/pytorch_gradcam"
                    ):
                        os.makedirs(
                            f"./{self.cfg.OUTPUT_DIR}/{self.cfg.MODEL.TRANSFER_TYPE}/pytorch_gradcam"
                        )

                    for i in range(attribution_patches.shape[0]):
                        # unique_id = str(uuid.uuid4())
                        unique_id = next(id_iter)
                        filename = f"./{self.cfg.OUTPUT_DIR}/{self.cfg.MODEL.TRANSFER_TYPE}/pytorch_gradcam/pytorch_gradcam_{unique_id}_{targets[i]}_{torch.argmax(outputs[i])}.png"

                        # attribution_patches = F.interpolate(attribution_patches, size=(224, 224), mode='bilinear', align_corners=False)

                        targetrgb = np.transpose(
                            X[i].squeeze().cpu().detach().numpy(), (1, 2, 0)
                        )

                        # print('attribution_patches', attribution_patches.shape) # torch.Size([128, 224, 224])
                        grayscale_cam = attribution_patches[i].cpu().detach().numpy()

                        # print('grayscale_cam', grayscale_cam.shape) # (224, 224)
                        # print('targetrgb', targetrgb.shape) # (224, 224, 3)

                        # print('1', min(grayscale_cam.flatten()), max(grayscale_cam.flatten())) # exist negative value
                        # print('2', min(targetrgb.flatten()), max(targetrgb.flatten()))

                        # print('1', np.max(grayscale_cam), np.min(grayscale_cam))

                        # grayscale_cam = np.interp(grayscale_cam, (min_val, max_val), (0, 255))

                        only_heatmap = True
                        if only_heatmap:
                            # targetrgb = cv2.cvtColor(targetrgb, cv2.COLOR_RGB2BGR)
                            # targetrgb = np.uint8(targetrgb)
                            heatmap = cv2.applyColorMap(
                                np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET
                            )
                            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                            # print('heatmap', heatmap.shape) # (224, 224, 3)
                            # print('targetrgb', targetrgb.shape) # (224, 224, 3)

                            # cv2.imwrite(f'{filename}.jpg', merged_img)

                            # fig, ax = plt.subplots(1, 2, figsize=(10,5))

                            # # plot the left image
                            # ax[0].imshow(targetrgb)
                            # ax[0].axis('off')
                            # # ax[0].set_title('Target RGB Image')

                            # # plot the right image
                            # ax[1].imshow(heatmap)
                            # ax[1].axis('off')
                            # # ax[1].set_title('Heatmap')
                            # plt.savefig(f'{filename}.jpg')

                            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                            ax.imshow(heatmap)
                            ax.axis("off")
                            plt.savefig(f"{filename}_heatmap.jpg")

                            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                            ax.imshow(targetrgb)
                            ax.axis("off")
                            plt.savefig(f"{filename}_original.jpg")

                        else:
                            original_targetrgb = cv2.cvtColor(
                                targetrgb, cv2.COLOR_RGB2BGR
                            )
                            original_targetrgb = np.uint8(original_targetrgb)
                            cv2.imwrite(f"{filename}_original.jpg", original_targetrgb)

                            targetrgb = (
                                targetrgb.astype(np.float32) / 255.0
                            )  # should convert to float32 and range to 0~1
                            merged_img = show_cam_on_image(
                                targetrgb, grayscale_cam
                            )  # use_rgb=True
                            cv2.imwrite(f"{filename}_heatmap.jpg", merged_img)

                        # targetrgb = np.uint8(targetrgb)
                        # merged_img = cv2.hconcat([targetrgb, heatmap])

                        # merged_img = show_cam_on_image(targetrgb, grayscale_cam, use_rgb=True)

                        # merged_img = cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)

            else:
                # print("attribution_patches is None")
                ValueError("attribution_patches is None")

            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1, total, losses.val, batch_time.val, data_time.val
                    )
                    + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)

    def forward_one_batch_IgGeneral(
        self, ig_patches, inputs, targets, is_train, integrated_method
    ):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape
                    )
                )

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights, self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(outputs, targets, self.cls_weights)

                # True
                batch_size = inputs.shape[0]
                num_batches = 64  # divide the attribution computation into 4 batches (32 for prompt/ 64 for finetune)
                chunk_size = batch_size // num_batches
                attributions = []
                for i in range(num_batches):
                    start_idx = i * chunk_size
                    end_idx = (
                        start_idx + chunk_size if i < num_batches - 1 else batch_size
                    )
                    inputs_chunk = inputs[start_idx:end_idx]
                    baseline_chunk = torch.zeros_like(inputs_chunk)
                    target_chunk = targets[start_idx:end_idx]

                    if target_chunk.shape[0] != 0:

                        # attribution_chunk = ig_patches.attribute(inputs_chunk, baselines=baseline_chunk, target=target_chunk)
                        if integrated_method == "pytorch_gradcam":

                            attribution_chunk = ig_patches(
                                input_tensor=inputs_chunk,
                                targets=None,
                                eigen_smooth=True,
                                aug_smooth=True,
                            )  # eigen_smooth=True, aug_smooth=False # target_chunk

                            # print(attribution_chunk.shape) # (1, 224, 224)
                            attribution_chunk = torch.from_numpy(attribution_chunk)
                            # Here grayscale_cam has only one image in the batch
                            # attribution_chunk = attribution_chunk[0, :]

                        else:
                            ValueError(
                                f"Unsupported cfg.ATTRIBUTION_INTEGRATED_METHOD in trainer.py forward_one_batch_IgGeneral: {integrated_method}"
                            )

                        attributions.append(attribution_chunk)
                        attribution_ig = torch.cat(attributions, dim=0)
                    else:
                        ValueError(
                            f"target_chunk.shape[0] == 0 in trainer.py forward_one_batch_IgGeneral: {target_chunk.shape[0]}"
                        )

            if loss == float("inf"):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs, attribution_ig
