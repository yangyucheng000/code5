from PIL import Image

# from .solver import Solver
from .solver_nolocald_nowarploss import Solver
from .preprocess import PreProcess


class Inference:
    """
    An inference wrapper for makeup transfer.
    It takes two image `source` and `reference` in,
    and transfers the makeup of reference to source.
    """
    def __init__(self, config, device="cpu", model_path="assets/models/G.pth"):
        """
        Args:
            device (str): Device type and index, such as "cpu" or "cuda:2".
            device_id (int): Specefying which devide index
                will be used for inference.
        """
        self.device = device
        self.solver = Solver(config, device, inference=model_path)
        self.preprocess = PreProcess(config, device)

    def transfer(self, source: Image, reference: Image, with_face=False):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """
        source_input, face, crop_face = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)
        if not (source_input and reference_input):
            if with_face:
                return None, None,None,None,None
            return

        for i in range(len(source_input)):
            source_input[i] = source_input[i]  # .to(self.device)

        for i in range(len(reference_input)):
            reference_input[i] = reference_input[i]  # .to(self.device)

        # TODO: Abridge the parameter list.
        result, imgs, mid_results = self.solver.test(*source_input, *reference_input)
        
        if with_face:
            return result, crop_face, imgs, source_input[4], reference_input[4], mid_results
        return result, mid_results 
