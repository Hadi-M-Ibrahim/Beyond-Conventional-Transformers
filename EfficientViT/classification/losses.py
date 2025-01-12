"""
Implements the knowledge distillation loss, proposed in deit
"""
import torch
from torch.nn import functional as F
import torch.nn as nn


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
            teacher_probs = torch.sigmoid(teacher_logits)


        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.binary_cross_entropy_with_logits(
                outputs_kd, teacher_probs
            )
        elif self.distillation_type == 'hard':
            teacher_hard_labels = (teacher_probs > 0.5).float()
            distillation_loss = F.binary_cross_entropy_with_logits(outputs_kd, teacher_hard_labels)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss