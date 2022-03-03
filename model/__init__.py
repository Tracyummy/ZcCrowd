# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build

def build_model(args):
    return build(args)


from . import CSRNet
from . import SANet
from . import transformer