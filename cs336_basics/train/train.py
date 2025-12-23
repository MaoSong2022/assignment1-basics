import time
import wandb
import argparse
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

import cs336_basics.model.model as transformer_model
from cs336_basics.train import optimizer as MyOptimizer
from cs336_basics.train import serialization
from cs336_basics.model import utils


class MemmapDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: int,
        stride: int = None,
        shuffle: bool = False,
        pad_id: int = 0,
        dtype=np.int32,
    ):
        self.file_path = file_path
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.pad_id = pad_id
        self.shuffle = shuffle

        self.memmap_arr = np.lib.format.open_memmap(file_path, mode="r", dtype=dtype)

        self.total_length = len(self.memmap_arr)

        self.sample_starts = []
        start = 0
        while start + self.seq_len <= self.total_length:
            self.sample_starts.append(start)
            start += self.stride

        # 处理最后一个不足seq_len的片段（填充）
        if start < self.total_length:
            self.sample_starts.append(start)

        self.num_samples = len(self.sample_starts)

        print(f"数据集加载完成：样本数={self.num_samples}，序列长度={self.total_length}")

        if self.shuffle:
            self.sample_indices = np.random.permutation(self.num_samples)
        else:
            self.sample_indices = np.arange(self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = self.sample_starts[self.sample_indices[idx]]
        end = start + self.seq_len
        token_ids = self.memmap_arr[start:end]

        if len(token_ids) < self.seq_len:
            pad_length = self.seq_len - len(token_ids)
            token_ids = np.pad(token_ids, (0, pad_length), mode="constant", constant_values=self.pad_id)

        return torch.tensor(token_ids, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--data_config_path", type=str)
    parser.add_argument("--train_config_path", type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Training on CPU. This will be extremely slow.")

    with open(args.model_config_path) as f:
        model_config = json.load(f)

    with open(args.data_config_path) as f:
        data_config = json.load(f)

    with open(args.train_config_path) as f:
        train_config = json.load(f)

    wandb.init(
        project=train_config["project_name"],
        name=train_config["run_name"],
        config={
            **model_config,
            **data_config,
            **train_config,
        },
    )

    # initialize model
    model = transformer_model.Model(**model_config)
    model.to(device)

    # initialize data set
    train_dataset = MemmapDataset(
        file_path=data_config["train_file_path"],
        seq_len=model_config["max_seq_len"],
        dtype=np.int32,
    )
    validation_dataset = MemmapDataset(
        file_path=data_config["validation_file_path"],
        seq_len=model_config["max_seq_len"],
        dtype=np.int32,
    )

    # initialize training stuff
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config["num_workers"],
        pin_memory=True,
    )

    optimizer = MyOptimizer.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=tuple(train_config["betas"]),
        weight_decay=train_config["weight_decay"],
        eps=train_config["eps"],
    )

    steps_per_epoch = len(train_loader)
    total_training_steps = train_config["epochs"] * steps_per_epoch
    print({"total_training_steps": total_training_steps})

    best_val_loss = float("inf")
    global_step = 0
    start_train_time = time.time()

    for epoch in range(train_config["epochs"]):
        epoch_start_time = time.time()  # record time used of current epoch
        model.train()

        train_loss = 0.0
        for batch_idx, batch_token_ids in enumerate(train_loader):
            batch_start_time = time.time()

            batch_token_ids = batch_token_ids.to(device)

            # update learning rate
            current_lr = MyOptimizer.get_lr_cosine_schedule(
                global_step,
                max_learning_rate=train_config["max_learning_rate"],
                min_learning_rate=train_config["min_learning_rate"],
                warmup_iters=train_config["warmup_iters"],
                cosine_cycle_iters=train_config["cosine_cycle_iters"],
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # forward pass
            logits = model(batch_token_ids)
            logits = logits[..., :-1, :].contiguous()
            labels = batch_token_ids[:, 1:].contiguous()
            labels = labels.view(-1)

            loss = utils.cross_entropy_loss(logits.view(-1, logits.size(-1)), labels)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            utils.gradient_clipping(
                model.parameters(),
                max_l2_norm=train_config["max_l2_norm"],
                eps=train_config["clipping_eps"],
            )
            optimizer.step()

            batch_time = time.time() - batch_start_time
            wandb.log(
                {
                    "train/batch_loss": loss.item(),  # 批次损失
                    "train/current_lr": current_lr,  # 当前学习率
                    "train/batch_time": batch_time,  # 批次耗时
                    "train/global_step": global_step,  # 全局步数
                    "train/epoch": epoch + 1,  # 当前epoch
                },
                step=global_step,
            )  # step指定为全局步数，保证可视化横轴对齐

            if batch_idx % 100 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                print(f"Train Batch {batch_idx} | LR: {current_lr:.6f} | Avg Loss: {avg_loss:.4f}")
            global_step += 1


            if global_step > 0 and global_step % train_config["eval_interval_steps"] == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_idx, batch_token_ids in enumerate(val_loader):
                        batch_token_ids = batch_token_ids.to(device)
                        # forward pass
                        logits = model(batch_token_ids)
                        # (batch_size, seq_len, vocab_size)
                        logits = logits[..., :-1, :].contiguous()
                        logits = logits.view(-1, logits.size(-1))
                        labels = batch_token_ids[:, 1:].contiguous().view(-1)
                        labels = batch_token_ids[:, 1:].contiguous()
                        labels = labels.view(-1)

                        loss = utils.cross_entropy_loss(logits, labels)

                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                wandb.log(
                    {
                        "val/interval_avg_loss": avg_val_loss,  # Epoch平均验证损失
                        "train/best_val_loss": best_val_loss,  # 最优验证损失
                    },
                    step=global_step,
                )

                # save the checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = "best_llm_model.pt"
                    serialization.save_checkpoint(model, optimizer, epoch, save_path)
                    print(f"最优模型已保存至: {save_path}")

                    # 记录最优模型信息
                    wandb.log(
                        {
                            "model/best_val_loss": best_val_loss,
                            "model/best_steps": epoch,
                            "model/save_path": save_path,
                        },
                        step=global_step,
                    )

        avg_train_loss = train_loss / len(train_loader)
        epoch_train_time = time.time() - epoch_start_time
        wandb.log({
            "train/epoch_avg_loss": avg_train_loss,  # Epoch平均训练损失
            "train/epoch_time": epoch_train_time,  # Epoch训练耗时
        })

    total_train_time = time.time() - start_train_time
    wandb.log(
        {
            "train/total_train_time": total_train_time,
            "train/total_train_time_hours": total_train_time / 3600,
        }
    )

    wandb.finish()
    print(f"\n训练完成！总耗时: {total_train_time / 3600:.2f} 小时")
    return model


if __name__ == "__main__":
    main()
