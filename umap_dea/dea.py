import logging

import numpy as np
from dealib.dea import RTS, Orientation, dea


logger = logging.getLogger(__name__)


def calculate_dea_for_embeddings(
    embeddings_df_dict: dict[str, np.ndarray],
    y: np.ndarray,
    rts: str = 'crs',
    orientation: str = 'input',
) -> dict[str, np.ndarray]:
    """Calculate DEA efficiency scores for each embedding."""

    if rts == 'crs':
        rts = RTS.crs
    elif rts == 'vrs':
        rts = RTS.vrs
    else:
        raise ValueError('rts must be either "crs" or "vrs"')
    orientation_str = orientation
    if orientation_str == 'input':
        orientation = Orientation.input
    elif orientation_str == 'output':
        orientation = Orientation.output
    else:
        raise ValueError('Orientation must be either "input" or "output"')
    efficiency_scores_dict = {}
    for embedding_name, embedding_df in embeddings_df_dict.items():
        logger.debug('Calculating DEA for embedding: %s...', embedding_name)
        eff = dea(
            embedding_df,
            y,
            rts=rts,
            orientation=orientation,
        ).eff
        # Guard against numerically unstable LP solutions that produce
        # impossible efficiency scores.
        if orientation_str == 'input':
            eff[eff > 1.0] = np.nan
        elif orientation_str == 'output':
            eff[eff < 1.0] = np.nan
        efficiency_scores_dict[embedding_name] = eff
    logger.info('DEA calculations completed.')
    return efficiency_scores_dict
