"""Multi-configuration (xyz) Point-JEPA variant.

Modes: pj_multi (three coordinate orderings, ordering_basis "coord" or "pca")
and pj_baseline (Point-JEPA's own single-ordering serialization).

"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from pointjepa.modules.EMA import EMA
from pointjepa.modules.point_sequencer import PointSequencer
from pointjepa.modules.masking import PointcloudMasking
from pointjepa.modules.target_sampler import TargetSampler
from pointjepa.modules.context_sampler import ContextSampler
from pointjepa.modules.tokenizer import PointcloudTokenizer
from pointjepa.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderOutput,
    TransformerPredictor,
)
from pointjepa.utils import transforms


class PointJepa(pl.LightningModule):
    def __init__(
        self,
        tokenizer_num_groups: int = 64,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.05,
        encoder_drop_path_rate: float = 0.25,
        encoder_add_pos_at_every_layer: bool = True,
        predictor_embed_dim: int = 192,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        predictor_mlp_ratio: float = 4.0,
        predictor_drop_out: float = 0.0,
        predictor_attention_dropout: float = 0.0,
        predictor_drop_path_rate: float = 0.0,
        predictor_add_target_pos: bool = False,
        predictor_add_pos_at_every_layer: bool = False,
        masking_mode: str = "pj_multi",  # "pj_multi" (3 orderings) | "pj_baseline" (Point-JEPA 1-view)
        ordering_mode: str = "xyz_axes",  # clean release supports only "xyz_axes"
        num_orderings: int = 3,
        ordering_basis: str = "coord",   # "coord" = coordinate axes; "pca" = principal axes
        ordering_axes: str = "",         # "" = first num_orderings of x,y,z; e.g. "xz" picks explicitly            # number of coordinate orderings (<= 3)
        token_seq_method: str = "iterative_nearest_min_start",
        target_sample_method: str = "contiguous",
        num_targets_per_sample: int = 4,
        target_sample_ratio: Tuple[float, float] = (0.15, 0.2),
        context_sample_method: str = "contiguous",
        context_sample_ratio: Optional[Tuple[float, float]] = (0.4, 0.75),
        target_layers: List[int] = [11],
        target_layer_part: str = "final",  # ffn, final
        target_layer_norm: Optional[str] = "layer",
        target_norm: Optional[str] = None,
        ema_tau_max: Optional[float] = 0.9998,
        ema_tau_min: Optional[float] = 0.99999,
        ema_tau_epochs: int = 200,
        loss: str = "smooth_l1",  # smooth_l1, mse
        loss_beta: float = 2,
        decorrelation_weight: float = 0.0,   # 0 = off; >0 adds covariance/decorrelation loss
        learning_rate: float = 1e-3,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 80,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        train_transformations: List[str] = [
            "subsample",
            "scale",
            "center",
            "unit_sphere",
            "rotate",
        ],  # subsample, scale, center, unit_sphere, rotate, translate, height_norm
        val_transformations: List[str] = ["subsample", "center", "unit_sphere"],
        transformation_subsample_points: int = 1024,
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
        transformation_rotate_dims: List[int] = [1],
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
        svm_validation: Dict[str, pl.LightningDataModule] = {},
        svm_validation_C=0.012,  # C=0.012 copied from Point-M2AE code
        fix_estimated_stepping_batches: Optional[int] = None,  # multi GPU bug fix
        pretrained_ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        def build_transformation(name: str) -> transforms.Transform:
            if name == "subsample":
                return transforms.PointcloudSubsampling(transformation_subsample_points)
            elif name == "scale":
                return transforms.PointcloudScaling(
                    min=transformation_scale_min, max=transformation_scale_max
                )
            elif name == "center":
                return transforms.PointcloudCentering()
            elif name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            elif name == "rotate":
                return transforms.PointcloudRotation(
                    dims=transformation_rotate_dims, deg=transformation_rotate_degs
                )
            elif name == "translate":
                return transforms.PointcloudTranslation(transformation_translate)
            elif name == "height_norm":
                return transforms.PointcloudHeightNormalization(
                    transformation_height_normalize_dim
                )
            elif name == "occlude":
                return transforms.PointcloudOcclusion()
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [build_transformation(name) for name in val_transformations]
        )

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )

        self.tokenizer = PointcloudTokenizer(
            num_groups=tokenizer_num_groups,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            token_dim=encoder_dim,
        )

        init_std = 0.02

        self.mask_token = nn.Parameter(torch.zeros(encoder_dim))
        nn.init.trunc_normal_(
            self.mask_token, mean=0, std=init_std, a=-init_std, b=init_std
        )

        dpr_encoder = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        dpr_predictor = [
            x.item() for x in torch.linspace(0, predictor_drop_path_rate, predictor_depth)
        ]
        self.student = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr_encoder,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        self.predictor = TransformerPredictor(
            embed_dim=encoder_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            mlp_ratio=predictor_mlp_ratio,
            qkv_bias=True,
            drop_rate=predictor_drop_out,
            attn_drop_rate=predictor_attention_dropout,
            drop_path_rate=predictor_drop_path_rate,
            add_target_pos=predictor_add_target_pos,
            add_pos_at_every_layer=predictor_add_pos_at_every_layer,
        )

        match loss:
            case "mse":
                self.loss_func = nn.MSELoss()
            case "smooth_l1":
                self.loss_func = nn.SmoothL1Loss(beta=loss_beta)
            case _:
                raise ValueError(f"Unknown loss: {loss}")

        if masking_mode in ("pj_multi", "pj_baseline"):
            # Point-JEPA's own min-coordinate ordering + its blocking (1-view baseline).
            self.target_sampler = TargetSampler(
                sample_method=target_sample_method,
                num_targets_per_sample=num_targets_per_sample,
                sample_ratio_range=target_sample_ratio,
            )
            self.context_sampler = ContextSampler(
                sample_method=context_sample_method,
                sample_ratio_range=context_sample_ratio,
            )
        if pretrained_ckpt_path is not None:
            self.load_state_dict(torch.load(pretrained_ckpt_path)["state_dict"], strict=False)
    @staticmethod
    def _cov_loss(z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, C) student context encoding -> decorrelate the C=384 dims
        z = z.reshape(-1, z.shape[-1]).float()
        z = z - z.mean(0, keepdim=True)
        N, C = z.shape
        cov = (z.T @ z) / (N - 1)
        off = cov - torch.diag(torch.diag(cov))
        return off.pow(2).sum() / C
    def setup(self, stage: Optional[str] = None) -> None:
        self.teacher = EMA(
            self.student,
            tau_min=(
                0
                if self.hparams.ema_tau_min is None  # type: ignore
                else self.hparams.ema_tau_min
            ),  # type: ignore
            tau_max=(
                1
                if self.hparams.ema_tau_max is None  # type: ignore
                else self.hparams.ema_tau_max
            ),  # type: ignore
            tau_steps=(
                self.hparams.fix_estimated_stepping_batches  # type: ignore
                or self.trainer.estimated_stepping_batches
            )
            * (self.hparams.ema_tau_epochs / self.trainer.max_epochs),  # type: ignore
            update_after_step=0,
            update_every=1,
        )

        svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        for dataset_name, datamodule in svm_validation.items():
            datamodule.setup("fit")
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.define_metric(
                        f"svm_train_acc_{dataset_name}", summary="last,max"
                    )
                    logger.experiment.define_metric(
                        f"svm_val_acc_{dataset_name}", summary="last,max"
                    )

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(self)

    @torch.no_grad()
    def _mincoord_order(self, centers: torch.Tensor) -> torch.Tensor:
        """Point-JEPA's own ordering: min-coordinate iterative-nearest -> (B,G) patch idx.
        Same sequence vanilla uses; lets us run a TRUE in-codebase vanilla baseline."""
        B, G, _ = centers.shape
        dev = centers.device
        seq = PointSequencer(method="iterative_nearest_min_start", device=dev)
        flat = seq.sort_iterative_nearest(centers, min_start=True)        # (B*G,)
        return flat.view(B, G) - (torch.arange(B, device=dev) * G).view(B, 1)        

    def _forward_pj_order(self, tokens, centers, order=None):
        B, G, C = tokens.shape
        assert order is not None, "clean release: pass an explicit ordering"
    
        # Reorder tokens and centers according to the Point-JEPA serialization.
        tok_r = torch.gather(
            tokens,
            1,
            order.unsqueeze(-1).expand(B, G, C),
        )
        cen_r = torch.gather(
            centers,
            1,
            order.unsqueeze(-1).expand(B, G, centers.shape[-1]),
        )
    
        self.target_sampler.setup_device(tokens.device)
        self.context_sampler.setup_device(tokens.device)
    
        # Adaptive PE (the paper's primary configuration): the shared positional
        # map is evaluated outside torch.no_grad(), so it is trained end-to-end.
        # The EMA teacher itself runs under @torch.no_grad() in generate_targets().
        pos_r = self.positional_encoding(cen_r)
        target_global = self.generate_targets(tok_r, pos_r)
        target_blocks, tgt_pos = self.target_sampler.sample(target_global)
    
        # Construct the context from the reordered student-side tokens.
        ctx_tokens, ctx_centers, _ = self.context_sampler.sample(
            tok_r,
            cen_r,
            tgt_pos,
        )
    
        # Student-side positional encoding remains differentiable.
        ctx_pos = self.positional_encoding(ctx_centers)
        context_encoding = self.student(
            ctx_tokens.half(),
            ctx_pos,
        ).last_hidden_state
    
        # Predict each target block from the shared context encoding.
        M = target_blocks.shape[0]
        prediction_blocks = torch.zeros_like(target_blocks)
    
        for i in range(M):
            tgt_centers = cen_r[:, tgt_pos[i], :]
    
            prediction_blocks[i] = self.predictor(
                context_encoding,
                ctx_centers,
                tgt_centers,
            )
    
        return prediction_blocks, target_blocks, context_encoding   
        
    def _forward_pj_multi(self, tokens, centers):
        B, G, E = tokens.shape
        Dc = centers.shape[-1]
        dev = tokens.device
        n = int(self.hparams.num_orderings)

        # Adaptive PE (the paper's primary configuration): the shared positional
        # map is evaluated outside torch.no_grad(), so it is trained end-to-end.
        # The EMA teacher runs once, under @torch.no_grad() in generate_targets(),
        # on the original token order; its targets are then reordered per ordering.
        pos = self.positional_encoding(centers)
        target_global = self.generate_targets(tokens, pos)                # (B,G,E) teacher ONCE

        Et = target_global.shape[-1]

        # all n orderings at once, then fold configs into the batch -> (B*n, G, .)
        if self.hparams.ordering_mode == "xyz_axes":
            # n fixed-direction orderings; folded into the batch below.
            assert n <= 3, "xyz_axes supports at most 3 orderings"
            _AX = {"x": 0, "y": 1, "z": 2}
            _spec = getattr(self.hparams, "ordering_axes", "") or ""
            _dims = [_AX[c] for c in _spec] if _spec else list(range(n))
            assert len(_dims) == n, f"ordering_axes {_spec!r} inconsistent with n={n}"
            _basis = getattr(self.hparams, "ordering_basis", "coord") or "coord"
            if _basis == "pca":
                _c = (centers - centers.mean(dim=1, keepdim=True)).float()      # (B,G,3)
                _cov = (_c.transpose(1, 2) @ _c) / max(_c.shape[1] - 1, 1)      # (B,3,3)
                _, _V = torch.linalg.eigh(_cov.float())    # eigh lacks a half CUDA kernel; compute in fp32
                _V = _V.to(_cov.dtype)                      # cast back (matches the trained runs)
                _V = _V.flip(-1)                                                # PC1,PC2,PC3 as columns
                _proj = _c @ _V                                                 # (B,G,3)
                orders = torch.stack([_proj[..., d].argsort(dim=-1) for d in _dims], dim=1)  # (B,n,G)
            else:
                orders = torch.stack([centers[..., d].argsort(dim=-1) for d in _dims], dim=1)  # (B,n,G)
        else:
            raise ValueError("this release supports only ordering_mode='xyz_axes'")
        def fold(src, D):
            idx = orders.unsqueeze(-1).expand(B, n, G, D)
            return torch.gather(src.unsqueeze(1).expand(B, n, G, D), 2, idx).reshape(B * n, G, D)
        tok_r = fold(tokens, E)                                            # (B*n, G, E)
        cen_r = fold(centers, Dc)                                          # (B*n, G, 3)
        tg_r  = fold(target_global, Et)                                    # (B*n, G, E)

        self.target_sampler.setup_device(dev)
        self.context_sampler.setup_device(dev)
        lo, hi = self.hparams.target_sample_ratio
        ratio = float(torch.empty((), device=dev).uniform_(lo, hi)) if hi > lo else float(lo)

        target_blocks, tgt_pos = self.target_sampler.select_embed_contiguous(tg_r, ratio)  # (M,B*n,Kt,E),(M,Kt)
        ctx_tokens, ctx_centers, _ = self.context_sampler.sample(tok_r, cen_r, tgt_pos)    # (B*n,Kc,E),(B*n,Kc,3)

        ctx_pos = self.positional_encoding(ctx_centers)
        context_encoding = self.student(ctx_tokens.half(), ctx_pos).last_hidden_state       # (B*n,Kc,E)

        M = target_blocks.shape[0]
        x = torch.zeros_like(target_blocks)                                # (M, B*n, Kt, E)
        for i in range(M):
            x[i] = self.predictor(context_encoding, ctx_centers, cen_r[:, tgt_pos[i], :])
        return x, target_blocks, context_encoding
        
    def _perform_step(self, inputs):
        # ONE FPS/grouping call so tokens, centers and groups share the same patches
        groups, centers, gidx = self.tokenizer.grouping(inputs, return_idx=True)             # (B,G,S,3),(B,G,3)
        Bx, Gx, Sx, Cx = groups.shape
        tokens = self.tokenizer.embedding(
            groups.reshape(Bx * Gx, Sx, Cx)
        ).reshape(Bx, Gx, self.tokenizer.token_dim)                   # (B,G,C)
        if self.hparams.masking_mode == "pj_baseline":
            return self._forward_pj_order(
                tokens, centers, order=self._mincoord_order(centers))
        if self.hparams.masking_mode == "pj_multi":
            return self._forward_pj_multi(tokens, centers)
        raise ValueError(f"Unknown masking_mode: {self.hparams.masking_mode}")
        
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
            # inputs: (B, N, 3)
            points, _ = batch
            points = self.train_transformations(points)
            x, y, ctx = self._perform_step(points)
            loss = self.loss_func(x, y)
            covl = self._cov_loss(ctx)
            self.log("train_cov_loss", covl, on_epoch=True)
            w = self.hparams.decorrelation_weight
            if w > 0:
                loss = loss + w * covl
            self.log("train_loss", loss, on_epoch=True)
            self.log("train_pred_std", self.token_std(x))  # should always be > 0.01
            self.log("train_target_std", self.token_std(y))  # should always be > 0.1
            return loss

    @torch.no_grad()
    def generate_targets(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Same as point2vec, but if target_layer=last, layer_part=final, target_layer_norm=layer, target_norm=none,
        then it should be the identical to IJEPA implementation, that is, teacher->take the final layer->layer norm.
        """
        assert self.teacher.ema_model is not None  # always false
        self.teacher.ema_model.eval()
        jepa_target_layers: List[int] = self.hparams.target_layers  # type: ignore
        jepa_target_layer_part: str = self.hparams.target_layer_part  # type: ignore
        output: TransformerEncoderOutput = self.teacher(
            tokens,
            pos,
            return_hidden_states=jepa_target_layer_part == "final",
            return_ffns=jepa_target_layer_part == "ffn",
        )
        if jepa_target_layer_part == "ffn":
            assert output.ffns is not None
            target_layers = output.ffns
        elif jepa_target_layer_part == "final":
            assert output.hidden_states is not None
            target_layers = output.hidden_states
        else:
            raise ValueError()
        target_layers = [target_layers[i] for i in jepa_target_layers]  # [(B, T, C)]

        # pre norm

        target_layer_norm = self.hparams.target_layer_norm  # type: ignore
        if target_layer_norm == "instance":
            target_layers = [
                F.instance_norm(target.transpose(1, 2)).transpose(1, 2)
                for target in target_layers
            ]
        elif target_layer_norm == "layer":
            target_layers = [
                F.layer_norm(target, target.shape[-1:]) for target in target_layers
            ]
        elif target_layer_norm == "group":
            target_layers = [
                F.layer_norm(target, target.shape[-2:]) for target in target_layers
            ]
        elif target_layer_norm == "batch":
            target_layers = [
                F.batch_norm(
                    target.transpose(1, 2),
                    running_mean=None,
                    running_var=None,
                    training=True,
                ).transpose(1, 2)
                for target in target_layers
            ]
        elif target_layer_norm is not None:
            raise ValueError()

        # Average top K blocks
        targets = torch.stack(target_layers, dim=0).mean(0)  # (B, T, C)

        # post norm

        target_norm = self.hparams.target_norm  # type: ignore
        if target_norm == "instance":
            targets = F.instance_norm(targets.transpose(1, 2)).transpose(1, 2)
        elif target_norm == "layer":
            targets = F.layer_norm(targets, targets.shape[-1:])
        elif target_norm == "group":
            targets = F.layer_norm(targets, targets.shape[-2:])
        elif target_norm == "batch":
            targets = F.batch_norm(
                targets.transpose(1, 2),
                running_mean=None,
                running_var=None,
                training=True,
            ).transpose(1, 2)
        elif target_norm is None:
            # Explicitly leaving a pass here to make it clear that we're not doing anything
            pass

        return targets

    def validation_step(
        self, batch, batch_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # inputs: (B, N, 3)
        points, _ = batch
        points = self.val_transformations(points)
        # validation_step:  (just unpack the extra value)
        x, y, ctx = self._perform_step(points)
        loss = self.loss_func(x, y)
        self.log("val_loss", loss)
        self.log("val_pred_std", self.token_std(x))
        self.log("val_target_std", self.token_std(y))

    def validation_epoch_end(
        self, outputs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        for dataset_name, datamodule in svm_validation.items():
            svm_train_acc, svm_val_acc, eff_eig = self.svm_validation(datamodule)
            self.log(f"svm_train_acc_{dataset_name}", svm_train_acc)
            self.log(f"svm_val_acc_{dataset_name}", svm_val_acc)
            self.log(f"eff_eig_{dataset_name}", eff_eig)
    def svm_validation(self, datamodule: pl.LightningDataModule) -> Tuple[float, float]:
        # Lightning controls the `training` and `grad_enabled` state. Don't want to mess with it, but make sure it's correct.
        assert not self.training
        assert not torch.is_grad_enabled()

        def xy(
            dataloader: DataLoader,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x_list = []
            label_list = []
            for points, label in iter(dataloader):
                points: torch.Tensor = points.cuda()
                label: torch.Tensor = label.cuda()
                points = self.val_transformations(points)
                embeddings, centers = self.tokenizer(points)
                pos = self.positional_encoding(centers)
                x = self.student(embeddings, pos).last_hidden_state
                x = torch.cat([x.max(dim=1).values, x.mean(dim=1)], dim=-1)
                x_list.append(x.cpu())
                label_list.append(label.cpu())

            x = torch.cat(x_list, dim=0)  # (N, 768)
            y = torch.cat(label_list, dim=0)  # (N,)
            return x, y

        x_train, y_train = xy(datamodule.train_dataloader())  # type: ignore
        x_val, y_val = xy(datamodule.val_dataloader())  # type: ignore

        svm_C: float = self.hparams.svm_validation_C  # type: ignore
        svm = SVC(C=svm_C, kernel="linear")
        svm.fit(x_train, y_train)  # type: ignore
        train_acc: float = svm.score(x_train, y_train)  # type: ignoref
        val_acc: float = svm.score(x_val, y_val)  # type: ignore
        # end of svm_validation, before return:
        Xv = x_val.numpy().astype(np.float64); Xv = Xv - Xv.mean(0, keepdims=True)
        Cov = (Xv.T @ Xv) / (len(Xv) - 1)
        eigv = np.clip(np.linalg.eigvalsh(Cov), 1e-12, None)
        eff_eig = float((eigv.sum()**2) / (eigv**2).sum())
        return train_acc, val_acc, eff_eig
        

    # https://github.com/Lightning-AI/lightning/issues/11688#issuecomment-1026688558
    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.teacher.update()
        self.log("ema_decay", self.teacher.get_current_decay())

    @staticmethod
    def token_std(tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, C)
        return tokens.reshape(-1, tokens.shape[-1]).std(0).mean()

    def configure_optimizers(self):
        assert self.trainer is not None

        if self.hparams.fix_estimated_stepping_batches is not None:  # type: ignore
            # check that the correct value for the multi GPU fix was provided
            assert self.trainer.estimated_stepping_batches == self.hparams.fix_estimated_stepping_batches  # type: ignore

        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.optimizer_adamw_weight_decay,  # type: ignore
        )

        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=self.hparams.lr_scheduler_linear_warmup_epochs,  # type: ignore
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=self.hparams.lr_scheduler_linear_warmup_start_lr,  # type: ignore
            eta_min=self.hparams.lr_scheduler_cosine_eta_min,  # type: ignore
        )

        return [opt], [sched]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # This is a bit of a hack. We want to avoid saving the datasets in the svm_validation dict,
        # as this would store the entire dataset inside the checkpoint, blowing it up to multiple GBs.
        checkpoint["hyper_parameters"]["svm_validation"] = {}
