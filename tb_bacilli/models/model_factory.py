"""
model_factory.py — Model creation with automatic fallback.

get_model(config):
- cavmamba: load + attempt pretrained weight loading
  if weight file missing → warn, continue with random init
  if mamba-ssm ImportError → auto-switch to switch_umamba
- switch_umamba: random init, no pretrained
- Print param summary
- Return model
"""

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_model(config):
    """
    Create model based on config.model_name.

    Args:
        config: TBConfig instance.

    Returns:
        Model instance.
    """
    model_name = config.model_name.lower()

    if model_name == "cavmamba":
        try:
            from .cavmamba import CaVMamba

            logger.info("Building CaVMamba model...")
            model = CaVMamba(
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                encoder_dims=config.encoder_dims,
                encoder_depths=config.encoder_depths,
                decoder_dims=config.decoder_dims,
                decoder_depths=config.decoder_depths,
                dropout=config.dropout,
                input_size=config.input_size,
            )

            # Attempt pretrained weight loading
            model.load_vmamba_pretrained(config.vmamba_pretrained_path)

        except ImportError as e:
            logger.warning(
                f"Failed to import CaVMamba dependencies: {e}. "
                f"Auto-switching to Switch-UMamba."
            )
            model_name = "switch_umamba"

    if model_name == "switch_umamba":
        from .switch_umamba import SwitchUMamba

        logger.info("Building Switch-UMamba model (from scratch, no pretrained)...")
        model = SwitchUMamba(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            encoder_dims=config.encoder_dims,
            encoder_depths=config.encoder_depths,
            dropout=config.dropout,
            input_size=config.input_size,
        )

    # Parameter summary
    params = count_parameters(model)
    logger.info(
        f"Model: {model_name} | "
        f"Total params: {params['total']:,} | "
        f"Trainable params: {params['trainable']:,}"
    )

    return model
