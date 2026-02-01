import torch

from src.config import DUMMY_INPUT_SHAPE
from src.models.mixnet_stub import MixNetStub
from src.models.mobilenet_stub import MobileNetStub


def run_pipeline():
    x = torch.randn(*DUMMY_INPUT_SHAPE)

    print("Input shape:", x.shape)

    model_mixnet = MixNetStub()
    out_mixnet = model_mixnet(x)
    print("MixNetStub output shape:", out_mixnet.shape)

    model_mobilenet = MobileNetStub()
    out_mobilenet = model_mobilenet(x)
    print("MobileNetStub output shape:", out_mobilenet.shape)


if __name__ == "__main__":
    run_pipeline()
