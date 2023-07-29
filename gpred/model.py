from typing import Dict, Tuple

import numpy as np

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torchvision


def compute_output_dim(layer, dim_input):
    """Compute the output dimensions for the given layer.

    Args:
        layer (torch.nn.Module): Torch layer.
        dim_input (tuple(int, ...)): Dimension of input.
    Returns:
        tuple[int, ...]: Dimension of output
    """
    # Linear: simple output size
    if type(layer) is nn.Linear:
        return (layer.out_features,)

    # BatchNorm: same dim
    if type(layer) in (nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU, nn.AdaptiveAvgPool2d):
        return dim_input

    if not type(layer) in (nn.Conv2d, nn.MaxPool2d):
        raise TypeError(f"Unsupported layer type {type(layer)}")

    # Conv2d or MaxPool2d
    hw_in = np.array(dim_input[-2:])
    padding = np.array(layer.padding)
    dilation = np.array(layer.dilation)
    kernel_size = np.array(layer.kernel_size)
    stride = np.array(layer.stride)

    hw_out = (hw_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    hw_out = tuple(hw_out.astype(int))

    # Conv2d: Replace in_channels with out_channels
    if type(layer) is nn.Conv2d and len(dim_input) > 2:
        return dim_input[:-3] + (layer.out_channels,) + hw_out

    return dim_input[:-2] + hw_out


class Net(nn.Module):
    """Network to predict symbolic state as a vector of proposition probabilities.

    The output is a vector of N logits, where N is the number of propositions in
    the symbolic state. The sigmoid of these logits represents the probabilities
    of the propositions. Logits are used for better numeric stability with the
    cross entropy loss.

    Args:
        dim_input (tuple(int, int, int)): Dimensions of input image (depth, height, width)
        dim_output (int): Number of propositions in the full symbolic state.
    """

    def __init__(self, dim_input):
        super().__init__()

        dim_input = tuple(dim_input[-3:])

        self._name_layers = ["input"]
        self._dims = [dim_input]
        self._params = [0]
        self._debug = False

    def _register_layer(self, name, layer):
        dim_input = self._dims[-1]
        self._name_layers.append(name)
        self._dims.append(compute_output_dim(layer, dim_input))
        if hasattr(layer, "weight"):
            self._params.append(layer.weight.numel() + layer.bias.numel())
        else:
            self._params.append(0)

    def _create_features(self, layers):
        # Attach features
        self.features = layers

        idx_feature = -1
        for i, layer in enumerate(self.features):
            if type(layer) is nn.Conv2d:
                idx_feature += 1
                self._register_layer(f"conv{idx_feature}", layer)
            elif type(layer) is nn.MaxPool2d:
                self._register_layer(f"pool{idx_feature}", layer)

        # Avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self._register_layer(f"avgpool{idx_feature+1}", self.avgpool)

        # Compute flattened feature size
        self._dim_flat = np.prod(self._dims[-1])

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor, (-1, 3, 32, 32)): Input image.
        Returns:
            torch.Tensor, (-1, N): Predicted proposition logits.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, self._dim_flat)
        x = self.classifier(x)
        return x


class SimplePropositionNet(Net):
    """Network to predict symbolic state as a vector of proposition probabilities.

    The output is a vector of N logits, where N is the number of propositions in
    the symbolic state. The sigmoid of these logits represents the probabilities
    of the propositions. Logits are used for better numeric stability with the
    cross entropy loss.

    Args:
        dim_input (tuple(int, int, int)): Dimensions of input image (depth, height, width)
        dim_output (int): Number of propositions in the full symbolic state.
        freeze_features (bool): Whether to freeze cnn features.
    """

    def __init__(self, dim_input, dim_output, freeze_features=False):
        super().__init__(dim_input)

        # vgg = torchvision.models.vgg11(pretrained=True)
        # if freeze_features:
        #     for param in vgg.parameters():
        #         param.requires_grad = False
        # self._create_features(vgg.features)

        resnet = torchvision.models.resnet18(pretrained=True)
        if freeze_features:
            for param in resnet.parameters():
                param.requires_grad = False
        resnet_features = list(resnet.children())[:-1]
        self.features = nn.Sequential(*resnet_features)
        self._dim_flat = resnet.fc.in_features

        self.classifier = nn.Linear(self._dim_flat, dim_output)
        # self._create_features(torchvision.models.vgg11(pretrained=True).features)

        # linear_layers = [
        #     nn.Linear(self._dim_flat, 4096),
        #     nn.Linear(4096, 4096),
        #     nn.Linear(4096, dim_output),
        # ]
        # self.classifier = nn.Sequential(
        #     linear_layers[0],
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     linear_layers[1],
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     linear_layers[2],
        # )
        # for i, layer in enumerate(linear_layers):
        #     self._register_layer(f"fc{i}", layer)
        #     nn.init.normal_(layer.weight, 0, 0.01)
        #     nn.init.constant_(layer.bias, 0)

        # for name, dim, param in zip(self._name_layers, self._dims, self._params):
        #     print(f"{name}: {dim}, {param}")

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor, (-1, 3, 32, 32)): Input image.
        Returns:
            torch.Tensor, (-1, N): Predicted proposition logits.
        """
        x = self.features(x)
        x = x.view(-1, self._dim_flat)
        x = self.classifier(x)
        return x


class StackedPropositionNet(Net):
    """Network to predict symbolic state as a vector of proposition probabilities.

    The output is a vector of N logits, where N is the number of propositions in
    the symbolic state. The sigmoid of these logits represents the probabilities
    of the propositions. Logits are used for better numeric stability with the
    cross entropy loss.

    Args:
        dim_input (tuple(int, int, int)): Dimensions of input image (depth, height, width)
        dim_output (int): Number of propositions in the full symbolic state.
    """

    def __init__(self, dim_input, dim_output):
        super().__init__(dim_input)

        self._create_features(torchvision.models.vgg11(pretrained=True).features)

        # linear_layers = [
        #     nn.Linear(self._dim_flat, 4096),
        #     nn.Linear(4096, 4096),
        #     nn.Linear(4096, 1),
        # ]
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self._dim_flat, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 1),
                )
                for _ in range(dim_output)
            ]
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor, (-1, 3, 32, 32)): Input image.
        Returns:
            torch.Tensor, (-1, N): Predicted proposition logits.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, self._dim_flat)
        xs = [classifier(x) for classifier in self.classifiers]
        x = torch.cat(xs, dim=1)
        return x


class BoundingBoxFeature(nn.Module):
    def __init__(self, num_objects, dim_per_object):
        super().__init__()
        self._num_objects = num_objects
        self._dim_per_object = dim_per_object
        dim_out = num_objects * dim_per_object

        self.conv_pos = nn.Conv2d(
            num_objects,
            dim_out,
            kernel_size=1,
            stride=1,
            bias=False,
            groups=num_objects,
        )
        self.conv_neg = nn.Conv2d(
            num_objects,
            dim_out,
            kernel_size=1,
            stride=1,
            bias=False,
            groups=num_objects,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """Forward pass.

        Args:
            x: [-1, num_objects, W, H] image.
        Returns:
            [-1, num_objects, dim_per_object, W, H] image.
        """
        out = self.conv_pos(x) + self.conv_neg(1 - x)
        out = out.view(
            out.shape[0], self._num_objects, self._dim_per_object, *out.shape[-2:]
        )

        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, downsample=False):
        super().__init__()
        # [7 x 7] => [7 x 7]
        self.conv1 = nn.Conv2d(dim_in, 512, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        # [7 x 7] => [7 x 7]
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        # [7 x 7] => [7 x 7]
        self.conv3 = nn.Conv2d(512, dim_out, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim_out)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if downsample:
            # [dim_in x 7 x 7] => [dim_out x 7 x 7]
            self.downsample = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(dim_out),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: [-1, 1, W, H] image.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, downsample=False):
        super().__init__()
        # [7 x 7] => [7 x 7]
        self.conv1 = nn.Conv2d(
            dim_in, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(512)

        # [7 x 7] => [7 x 7]
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if downsample:
            # [dim_in x 7 x 7] => [dim_out x 7 x 7]
            self.downsample = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(dim_out),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: [-1, 1, W, H] image.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PropositionNet(Net):
    """Network to predict symbolic state as a vector of proposition probabilities.

    The output is a vector of N logits, where N is the number of propositions in
    the symbolic state. The sigmoid of these logits represents the probabilities
    of the propositions. Logits are used for better numeric stability with the
    cross entropy loss.

    Args:
        dim_input (tuple(int, int, int)): Dimensions of input image (depth, height, width)
        dim_output (int): Number of propositions in the full symbolic state.
        model: 'resnet' or 'vgg'.
        freeze_features (bool): Whether to freeze cnn features.
    """

    def __init__(
        self, dim_input, num_propositions, model: str = "resnet", freeze_features=True
    ):
        super().__init__(dim_input)
        NUM_OBJECTS = 4

        # print(f"Loading PropositionNet with {model}.")
        if model == "resnet":
            resnet = torchvision.models.resnet50(pretrained=True)
            if freeze_features:
                for param in resnet.parameters():
                    param.requires_grad = False

            # First four layers of ResNet50
            resnet50_c4 = list(resnet.children())[:-3]  # Output of conv4
            self.image_features = nn.Sequential(*resnet50_c4)

            DIM_PER_OBJECT = 256
            self.bbox_features = BoundingBoxFeature(
                num_objects=NUM_OBJECTS, dim_per_object=DIM_PER_OBJECT
            )

            # Fifth layer of ResNet50
            self.features = nn.Sequential(
                ResidualBlock(
                    1024 + NUM_OBJECTS * DIM_PER_OBJECT, 2048, downsample=True
                ),
                ResidualBlock(2048, 2048),
                ResidualBlock(2048, 2048),
                nn.AdaptiveAvgPool2d(output_size=1),
            )

            # Predicate classification
            self.classifier = nn.Linear(in_features=2048, out_features=num_propositions)
        else:
            vgg = torchvision.models.vgg19_bn(pretrained=True)
            if freeze_features:
                for param in vgg.parameters():
                    param.requires_grad = False

            # VGG features.
            self.image_features = vgg.features

            DIM_PER_OBJECT = 128
            self.bbox_features = BoundingBoxFeature(
                num_objects=NUM_OBJECTS, dim_per_object=DIM_PER_OBJECT
            )

            # Fifth layer of ResNet50
            self.features = nn.Sequential(
                ResidualBlock(
                    512 + NUM_OBJECTS * DIM_PER_OBJECT, 2048, downsample=True
                ),
                ResidualBlock(2048, 2048),
                ResidualBlock(2048, 2048),
                nn.AdaptiveAvgPool2d(output_size=1),
            )

            # Predicate classification
            self.classifier = nn.Linear(in_features=2048, out_features=num_propositions)

        self._debug_values: Dict[str, np.ndarray] = {}

    def forward(self, x_boxes: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            x_boxes: (x, boxes) pair. x has shape [-1, 3 + num_objects, H, W],
                     where the object channels are binary masks over the
                     bounding box of the corresponding argument. boxes has shape
                     [-1, num_objects, 5], where the last dimension contains
                     the 5-tuple (idx_batch, x1, y1, x2, y2).
        Returns:
            Predicted predicate logits as a [-1, num_propositions] tensor.
        """
        x, boxes = x_boxes

        # Extract rgb image.
        # [-1, 7, 240, 494] => [-1, 3, 240, 494]
        x_i = x[:, :3, :, :]

        # Extract box mask image.
        # [-1, 7, 240, 494] => [-1, 4, 240, 494]
        x_o = x[:, 3:, :, :]

        # num_objects = x_o.shape[1]

        # Extract image box.
        # [-1, 5, 5] => [-1, 5] (idx_batch, x1, y1, x2, y2)
        boxes_i = boxes[:, 0, :]

        # Get image features from ResNet.
        # [-1, 3, 240, 494] => [-1, 1024, 15, 31]
        f_i = self.image_features(x_i)

        # Get roi features for image.
        # [-1, 1024, 15, 31] => [-1, 1024, 7, 7]
        f_ir = torchvision.ops.roi_align(
            f_i, boxes_i, output_size=7, spatial_scale=f_i.shape[2] / x_i.shape[2]
        )

        # [-1, 4, 240, 494] => [-1, 4, 7, 7]
        f_or = torchvision.ops.roi_align(x_o, boxes_i, output_size=7)

        # Get object features from mask image.
        # [-1, 4, 7, 7] => [-1, 4, 256, 7, 7]
        f_ob = self.bbox_features(f_or)

        # [-1, 4, 256, 7, 7] => [-1, 1024, 7, 7]
        f_obf = f_ob.view(-1, f_ob.shape[1] * f_ob.shape[2], *f_ob.shape[3:])

        # Concatenate channels.
        # [-1, 2048, 7, 7]
        f_io = torch.cat((f_ir, f_obf), dim=1)

        # [-1, 2048, 7, 7] => [-1, 2048]
        f = torch.squeeze(self.features(f_io))

        # [-1, 2048] => [-1, num_propositions]
        out = self.classifier(f)

        if self._debug:
            self._debug_values["x_i"] = x_i.cpu()
            self._debug_values["x_o"] = x_o.cpu()
            self._debug_values["boxes_i"] = boxes_i.cpu()
            self._debug_values["f_i"] = f_i.cpu()
            self._debug_values["f_ir"] = f_ir.cpu()
            self._debug_values["f_or"] = f_or.cpu()
            self._debug_values["f_ob"] = f_ob.cpu()
            self._debug_values["f_obf"] = f_obf.cpu()
            self._debug_values["f_io"] = f_io.cpu()
            self._debug_values["f"] = f.cpu()
            self._debug_values["out"] = out.cpu()

        return out


class PredicateNet(Net):
    """Network to predict symbolic state as a vector of predicate probabilities.

    The output is a vector of N logits, where N is the number of propositions in
    the symbolic state. The sigmoid of these logits represents the probabilities
    of the propositions. Logits are used for better numeric stability with the
    cross entropy loss.

    Args:
        dim_input: Dimensions of input image (rgb + num_objects, height, width).
        num_predicates: Number of predicates in the full symbolic state.
        num_args: Number of predicate arguments.
        model: 'resnet' or 'vgg'.
        freeze_features: Whether to freeze cnn features.
    """

    def __init__(
        self,
        dim_input: Tuple[int, int, int],
        num_predicates: int,
        num_args: int,
        model: str = "resnet",
        freeze_features: bool = True,
    ):
        super().__init__(dim_input)
        NUM_OBJECTS = num_args
        self._num_objects = num_args

        # print(f"Loading PredicateNet with {model}.")
        if model == "resnet":
            resnet = torchvision.models.resnet50(pretrained=True)
            if freeze_features:
                for param in resnet.parameters():
                    param.requires_grad = False

            # First four layers of ResNet50
            resnet50_c4 = list(resnet.children())[:-3]  # Output of conv4
            self.image_features = nn.Sequential(*resnet50_c4)

            DIM_PER_OBJECT = 256
            self._dim_per_object = DIM_PER_OBJECT

            # Fifth layer of ResNet50
            self.features = nn.Sequential(
                # ResidualBlock(
                #     1024 + NUM_OBJECTS * DIM_PER_OBJECT, 2048, downsample=True,
                # ),
                ResidualBlock(
                    (1 + NUM_OBJECTS) * 1024 + NUM_OBJECTS * DIM_PER_OBJECT,
                    2048,
                    downsample=True,
                ),
                ResidualBlock(2048, 2048),
                ResidualBlock(2048, 2048),
                nn.AdaptiveAvgPool2d(output_size=1),
            )

            # Fifth layer of ResNet18
            # self.obj_features = nn.Sequential(
            #     BasicBlock(1024, 512, downsample=True,),
            #     BasicBlock(512, 512),
            #     nn.AdaptiveAvgPool2d(output_size=1),
            # )

            # Predicate classification
            self.classifier = nn.Linear(in_features=2048, out_features=num_predicates)
            # self.classifier = nn.Linear(
            #     in_features=2048 + self._num_objects * 512, out_features=num_predicates
            # )
        else:
            vgg = torchvision.models.vgg19_bn(pretrained=True)
            if freeze_features:
                for param in vgg.parameters():
                    param.requires_grad = False

            # VGG features.
            self.image_features = vgg.features

            DIM_PER_OBJECT = 128
            self.bbox_features = BoundingBoxFeature(
                num_objects=NUM_OBJECTS, dim_per_object=DIM_PER_OBJECT
            )

            # Fifth layer of ResNet50
            self.features = nn.Sequential(
                ResidualBlock(
                    512 + NUM_OBJECTS * DIM_PER_OBJECT, 2048, downsample=True
                ),
                ResidualBlock(2048, 2048),
                ResidualBlock(2048, 2048),
                nn.AdaptiveAvgPool2d(output_size=1),
            )

            # Predicate classification
            self.classifier = nn.Linear(in_features=2048, out_features=num_predicates)

        self._debug_values: Dict[str, np.ndarray] = {}

    def forward(
        self, img_masks_boxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            img_boxes: (img, boxes) pair. img has shape [-1, 3 + num_objects, H, W],
                     where the object channels are binary masks over the
                     bounding box of the corresponding argument. boxes has shape
                     [-1, 1 + num_objects, 5], where the last dimension contains
                     the 5-tuple (idx_batch, x1, y1, x2, y2).
        Returns:
            Predicted predicate logits as a [-1, num_predicates] tensor.
        """
        img_rgb, img_mask, boxes = img_masks_boxes

        # Extract image box.
        # [-1, 1 + O, 5] => [-1, 5] (batch, idx/x1/y1/x2/y2)
        box_roi = boxes[:, 0, :]

        # [-1, 1 + O, 5] => [-O, 5] (batch * obj, idx/x1/y1/x2/y2)
        box_obj = boxes[:, 1:, :].reshape((-1, boxes.shape[-1]))

        # Get image features from ResNet.
        # [-1, 3, 240, 494] => [-1, 1024, 15, 31]
        f_rgb = self.image_features(img_rgb)
        H = img_rgb.shape[2]
        H_f = f_rgb.shape[2]

        # Get roi features for image.
        # [-1, 1024, 15, 31] => [-1, 1024, 7, 7]
        f_rgb_roi = torchvision.ops.roi_align(
            f_rgb, box_roi, output_size=7, spatial_scale=H_f / H
        )

        # [-1, 1024, 15, 31] => [-O, 1024, 7, 7]
        f_rgb_obj = torchvision.ops.roi_align(
            f_rgb, box_obj, output_size=7, spatial_scale=H_f / H
        )
        # [-O, 1024, 7, 7] => [-1, O * 1024, 7, 7]
        f_rgb_obj = f_rgb_obj.reshape(
            (-1, self._num_objects * f_rgb_obj.shape[1], *f_rgb_obj.shape[2:])
        )

        # [-1, O, 240, 494] => [-1, O, 7, 7]
        img_mask_roi = torchvision.ops.roi_align(img_mask, box_roi, output_size=7)

        # [-1, O, 7, 7] => [-1, O * DIM_PER_OBJ, 7, 7]
        f_mask_roi = torch.repeat_interleave(img_mask_roi, self._dim_per_object, dim=1)

        # Concatenate channels.
        # [-1, (1 + O) * 1024 + O * DIM_PER_OBJ, 7, 7]
        f_rgb_mask = torch.cat((f_rgb_roi, f_mask_roi, f_rgb_obj), dim=1)

        # [-1, 1024 + O * DIM_PER_OBJ, 7, 7] => [-1, 2048]
        f_roi = torch.squeeze(self.features(f_rgb_mask))

        # [-1, 2048] => [-1, num_predicates]
        out = self.classifier(f_roi)

        if self._debug:
            self._debug_values = {}
            self._debug_values["img_rgb"] = img_rgb.cpu()
            self._debug_values["img_mask"] = img_mask.cpu()
            self._debug_values["box_roi"] = box_roi.cpu()
            self._debug_values["box_obj"] = box_obj.cpu()
            self._debug_values["f_rgb"] = f_rgb.cpu()
            self._debug_values["f_rgb_roi"] = f_rgb_roi.cpu()
            self._debug_values["f_rgb_obj"] = f_rgb_obj.cpu()
            self._debug_values["img_mask_roi"] = img_mask_roi.cpu()
            self._debug_values["f_mask_roi"] = f_mask_roi.cpu()
            self._debug_values["f_rgb_mask"] = f_rgb_mask.cpu()
            self._debug_values["f_roi"] = f_roi.cpu()
            self._debug_values["out"] = out.cpu()

        # del box_roi
        # del box_obj
        # del f_rgb
        # del f_rgb_roi
        # del f_rgb_obj
        # del img_mask_roi
        # del f_mask_roi
        # del f_rgb_mask
        # del f_roi

        return out
