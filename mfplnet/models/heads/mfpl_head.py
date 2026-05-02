import math

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mfplnet.utils.lane import Lane
from mfplnet.models.losses.focal_loss import FocalLoss
from mfplnet.models.losses.accuracy import accuracy
from mfplnet.ops import nms

from mfplnet.models.utils.roi_gather import ROIGather, LinearModule
from mfplnet.models.utils.seg_decoder import SegDecoder
from mfplnet.models.utils.dynamic_assign import assign
from mfplnet.models.losses.lineiou_loss import liou_loss
from ..registry import HEADS


@HEADS.register_module
class MFPLHead(nn.Module):
    def __init__(self,
                 num_points=72,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=192,
                 num_fc=2,
                 refine_layers=3,
                 sample_points=108,
                 cfg=None):
        super(MFPLHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1,
                                       0,
                                       steps=self.n_offsets,
                                       dtype=torch.float32))

        self.prior_feat_channels = prior_feat_channels

        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)

        self.seg_decoder = SegDecoder(self.img_h, self.img_w,
                                      self.cfg.num_classes,
                                      self.prior_feat_channels,
                                      self.refine_layers)

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)

        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 +
            1)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                     weight=weights)

        self.init_weights()

    def init_weights(self):
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)

        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.

        grid = torch.cat((prior_xs, prior_ys), dim=-1)

        feature = F.grid_sample(
            batch_features,
            grid,
            align_corners=True
        )

        feature = feature.permute(0, 2, 1, 3)

        feature = feature.reshape(
            batch_size * num_priors,
            self.prior_feat_channels,
            self.sample_points,
            1
        )

        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight

        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets),
            device=predictions.device
        )

        priors[:, 2:5] = predictions.clone()

        priors[:, 6:] = (
                            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) * (self.img_w - 1) +
                            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
                              priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets))
                             * self.img_h /
                             torch.tan(
                                 priors[:, 4].unsqueeze(1).clone().repeat(1, self.n_offsets) * math.pi + 1e-5))
                        ) / (self.img_w - 1)

        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]

        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        left_num = self.num_priors * 3 // 8
        bottom_num = self.num_priors // 4
        right_num = self.num_priors * 3 // 8

        left_strip_size = 1 / (left_num // 4 + 1)
        for i in range(left_num):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], (i // 4 + 1) * left_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            if i % 4 == 0:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], -0.3)
            elif i % 4 == 1:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], -0.1)
            elif i % 4 == 2:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.1)
            else:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.3)


        bottom_strip_size = 1 / (bottom_num // 4 + 1)
        for i in range(left_num, left_num + bottom_num):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 1],
                ((i - left_num) // 4 + 1) * bottom_strip_size
            )
            if i % 4 == 0:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.2)
            elif i % 4 == 1:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.4)
            elif i % 4 == 2:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.6)
            else:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.8)

        right_strip_size = 1 / (right_num // 4 + 1)
        for i in range(left_num + bottom_num, self.num_priors):
            idx = i - (left_num + bottom_num)
            nn.init.constant_(self.prior_embeddings.weight[i, 0], (idx // 4 + 1) * right_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            if i % 4 == 0:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.7)
            elif i % 4 == 1:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.85)
            elif i % 4 == 2:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 1.15)
            else:
                nn.init.constant_(self.prior_embeddings.weight[i, 2], 1.3)

    def forward(self, x, **kwargs):
        batch_features = list(x[len(x) - self.refine_layers:])

        batch_features.reverse()

        batch_size = batch_features[-1].shape[0]

        if self.training:
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        priors, priors_on_featmap = self.priors.repeat(batch_size, 1, 1), self.priors_on_featmap.repeat(batch_size, 1,
                                                                                                        1)

        predictions_lists = []

        prior_features_stages = []
        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]

            prior_xs = torch.flip(priors_on_featmap, dims=[2])

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages, batch_features[stage], stage)

            fc_features = fc_features.view(num_priors, batch_size, -1).reshape(
                batch_size * num_priors, self.fc_hidden_dim)

            cls_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            cls_logits = self.cls_layers(cls_features)

            reg_features = fc_features.clone()
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])
            reg = reg.reshape(batch_size, -1, reg.shape[1])

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits
            predictions[:, :, 2:5] += reg[:, :, :3]
            predictions[:, :, 5] = reg[:, :, 3]

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            predictions[..., 6:] = (
                                           tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                                           ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                                             tran_tensor(predictions[..., 2])) * self.img_h /
                                            torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))
                                   ) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]
            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        if self.training:
            seg = None
            seg_features = torch.cat([
                F.interpolate(feature,
                              size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                              mode='bilinear',
                              align_corners=False)
                for feature in batch_features
            ], dim=1)
            seg = self.seg_decoder(seg_features)
            output = {'predictions_lists': predictions_lists, 'seg': seg}
            return self.loss(output, kwargs['batch'])

        return predictions_lists[-1]

    def predictions_to_pred(self, predictions):
        self.prior_ys = self.prior_ys.to(predictions.device).double()

        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]

            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)

            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)

            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))

            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2

            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]

            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                       self.cfg.cut_height) / self.cfg.ori_img_h

            if len(lane_xs) <= 1:
                continue

            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)

            lane = Lane(
                points=points.cpu().numpy(),
                metadata={
                    'start_x': lane[3],
                    'start_y': lane[2],
                    'conf': lane[1]
                })
            lanes.append(lane)

        return lanes


    def loss(self, output, batch, cls_loss_weight=2., xyt_loss_weight=0.5,
             iou_loss_weight=2., seg_loss_weight=1.):
        if self.cfg.haskey('cls_loss_weight'):
            cls_loss_weight = self.cfg.cls_loss_weight
        if self.cfg.haskey('xyt_loss_weight'):
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if self.cfg.haskey('iou_loss_weight'):
            iou_loss_weight = self.cfg.iou_loss_weight
        if self.cfg.haskey('seg_loss_weight'):
            seg_loss_weight = self.cfg.seg_loss_weight

        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss = 0
        reg_xytl_loss = 0
        iou_loss = 0
        cls_acc = []

        for stage in range(self.refine_layers):
            cls_acc_stage = []
            predictions_list = predictions_lists[stage]

            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss += cls_criterion(cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h)

                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= (self.img_w - 1)
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()
                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180

                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()

                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        (predictions[matched_row_inds, 2] * self.n_strips).round().long(), 0, self.n_strips)
                    target_starts = (target[matched_col_inds, 2] * self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts)

                cls_loss += cls_criterion(cls_pred, cls_target).sum() / target.shape[0]

                reg_xytl_loss += F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean()

                iou_loss += liou_loss(reg_pred, reg_targets, self.img_w, length=15)

                cls_acc_stage.append(accuracy(cls_pred, cls_target))

            if len(cls_acc_stage) > 0:
                cls_acc.append(sum(cls_acc_stage) / len(cls_acc_stage))
            else:
                cls_acc.append(0.0)

        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1), batch['seg'].long())

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
               + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

        return_value = {
            'loss': loss,
            'loss_stats': {
                'loss': loss,
                'cls_loss': cls_loss * cls_loss_weight,
                'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
                'seg_loss': seg_loss * seg_loss_weight,
                'iou_loss': iou_loss * iou_loss_weight
            }
        }

        for i in range(self.refine_layers):
            return_value['loss_stats'][f'stage_{i}_acc'] = cls_acc[i]

        return return_value



    def get_lanes(self, output, as_lanes=True):
        softmax = nn.Softmax(dim=1)

        decoded = []
        for predictions in output:
            threshold = self.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]

            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            nms_predictions = predictions.detach().clone()
            nms_predictions = torch.cat([nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[..., 5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores,
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes
            )
            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)

            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded