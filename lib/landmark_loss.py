import torch.nn as nn
import face_alignment
from torchvision import transforms
import torch

class LandmarkLoss(nn.Module):
    def __init__(self, use_cuda=True):
        super(LandmarkLoss, self).__init__()
        self.use_cuda = use_cuda
            
        self.landmark_ext_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda' if use_cuda else 'cpu')


        # TODO: add comment
        self.mse_loss = nn.MSELoss()

    # TODO write description
    def forward(self, y_hat, y):
        """

        Args:
            y_hat (torch.Tensor):
            y (torch.Tensor):

        Returns:

        """
        # Check if it returns None, skip if it does.
        y_hat_landmarks = torch.Tensor(self.landmark_ext_model.get_landmarks_from_image(y_hat))
        y_landmarks = torch.Tensor(self.landmark_ext_model.get_landmarks_from_image(y))
        if y_hat_landmarks == None or y_landmarks == None:
            return None
        else:
            landmarks_loss = self.mse_loss(y_hat_landmarks, y_landmarks)
            return landmarks_loss
