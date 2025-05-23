import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF


def dynamic_label_weights(inputs, outputs, teacher_probs, labels):
    weights = 1.0 - teacher_probs.mean(dim=0)
    return weights

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.

    It supports label-specific adjustments via the optional `label_weights` parameter.
    If provided, `label_weights` can either be:
      - A Tensor (typically 1D with length equal to the number of labels) that scales the loss.
      - A callable that returns a Tensor of label weights dynamically for each batch.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, label_weights=dynamic_label_weights):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.label_weights = label_weights  

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are fed to the teacher model.
            outputs: The outputs of the student model. It is expected to be either a Tensor,
                     or a Tuple[Tensor, Tensor] where the second element is the distillation output.
            labels: The labels for the base criterion.
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model must return a "
                             "tuple with the base output and the distillation output.")

        with torch.no_grad():
            teacher_inputs = TF.rgb_to_grayscale(inputs, num_output_channels=1)
            teacher_logits = self.teacher_model(teacher_inputs)
            teacher_probs = torch.sigmoid(teacher_logits)

        if self.label_weights is not None:
            if callable(self.label_weights):
                lw = self.label_weights(inputs, outputs, teacher_probs, labels)
            else:
                lw = self.label_weights

            if lw.dim() == 1:
                lw = lw.unsqueeze(0).expand_as(outputs_kd)
        else:
            lw = None

        if self.distillation_type == 'soft':
            distillation_loss = F.binary_cross_entropy_with_logits(
                outputs_kd, teacher_probs, weight=lw
            )
        elif self.distillation_type == 'hard':
            teacher_hard_labels = (teacher_probs > 0.5).float()
            distillation_loss = F.binary_cross_entropy_with_logits(
                outputs_kd, teacher_hard_labels, weight=lw
            )
    
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
