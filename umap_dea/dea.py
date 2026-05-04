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
    if orientation == 'input':
        orientation = Orientation.input
    elif orientation == 'output':
        orientation = Orientation.output
    else:
        raise ValueError('Orientation must be either "input" or "output"')
    efficiency_scores_dict = {}
    for embedding_name, embedding_df in embeddings_df_dict.items():
        logger.info('Calculating DEA for embedding: %s...', embedding_name)
        efficiency_scores_dict[embedding_name] = dea(
            embedding_df,
            y,
            rts=rts,
            orientation=orientation,
        ).eff
    logger.info('DEA calculations completed.')
    return efficiency_scores_dict
