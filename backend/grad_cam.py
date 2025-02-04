from PIL import Image
import numpy as np
import torch


class CamExtractor:
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer, output_layer):
        self.model = model
        self.target_layer = target_layer
        self.output_layer = output_layer
        self.gradients = None
        self.activations = None
        self.output_activations = None

    def save_gradient(self, module, input, grad):
        self.gradients = input

    def save_activations(self, module, input, act):
        self.activations = act

    def save_output_activations(self, module, input, outp):
        self.output_activations = outp

    def forward_pass(self, inp):
        # register hooks needed
        for n, m in self.model.named_modules():
            if n == self.target_layer:
                m.register_backward_hook(self.save_gradient)
                m.register_forward_hook(self.save_activations)
            if n == self.output_layer:
                m.register_forward_hook(self.save_output_activations)
        # forward pass to trigger hooks used
        t = self.model(inp)
        return self.activations, self.output_activations

    #
    # def forward_pass_on_convolutions(self, x):
    #     """
    #         Does a forward pass on convolutions, hooks the function at given layer
    #     """
    #     conv_output = None
    #     for module_pos, module in self.model.features._modules.items():
    #         x = module(x)  # Forward
    #         if int(module_pos) == self.target_layer:
    #             x.register_hook(self.save_gradient)
    #             conv_output = x  # Save the convolution output on that layer
    #     return conv_output, x
    #
    # def forward_pass(self, x):
    #     """
    #         Does a full forward pass on the model
    #     """
    #     # Forward pass on the convolutions
    #     conv_output, x = self.forward_pass_on_convolutions(x)
    #     x = x.view(x.size(0), -1)  # Flatten
    #     # Forward pass on the classifier
    #     x = self.model.classifier(x)
    #     return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, conv_act_layer, output_layer=''):
        self.model = model
        self.model.eval()
        if output_layer == '':
            names = []
            for n, _ in self.model.named_modules():
                names.append(n)
            output_layer = names[-1]
        # Define extractor
        self.extractor = CamExtractor(self.model, conv_act_layer, output_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(model_output.size()).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.cuda()
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients[0].data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam