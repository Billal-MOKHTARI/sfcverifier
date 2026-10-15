"""Runtime structural verification of SFC executions with the fired-transition matrix Phi."""

from .model import SFCModel
from .verifier import TransitionCheck, TransitionVerifier, VerifierConfig, to_mask

__all__ = ["SFCModel", "TransitionVerifier", "TransitionCheck", "VerifierConfig", "to_mask"]
__version__ = "1.0.0"
