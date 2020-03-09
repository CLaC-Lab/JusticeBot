"""General utility functions"""
def get_the_best_split(candidate_indices):
    """
    returns best split between facts and analysis sections
    in : list of integers for candidate indices such as [1,1,0,1,0] for instance
    out : integer, position of the last paragraph in be included in the fact section
    """

    if len(candidate_indices) == 1: 
        return 0
    candidate_splits = {}
    for split in range(1, len(candidate_indices)):
        facts_length = split/len(candidate_indices)
        analysis_side = candidate_indices[split:]
        analysis_purity = analysis_side.count(1)/len(analysis_side)
        square_difference = (facts_length - analysis_purity) ** 2
        candidate_splits.update({split:square_difference})

    return max(candidate_splits, key=candidate_splits.get)
