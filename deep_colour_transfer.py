'''
implement deep colour transfer using histogram analogy
following this paper: https://link.springer.com/epdf/10.1007/s00371-020-01921-6?sharing_token=m2UzXwVlSCP8CbRYNrEcnve4RwlQNchNByi7wbcMAY5_mQV2iPdNT8_ORizvbX3p8mina4UHEjoKsvegf0S_FwC3Yo3cBRV6mlx1mdbvv3CiiREpz3ZqyJuRGmHbygkNL_7X-3hd2CMGSxgPtF22LPsyjpEfhG1R_bNHSSVNvbc=
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.Autograd import Variable

class ColourTransferModel():
    def __init__(self) -> None:
        pass