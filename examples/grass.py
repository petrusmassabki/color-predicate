from src.colorpredicate import ColorPredicate

hist_args = {
    'color_space': 'hsv',
    'ch_indexes': (0, 1, 2),
    'bins': (8, 8, 8),
    'target_sr': 1.0
}

gauss_hist_args = {
    't_amp': 1.0,
    't_cov': 0.01,
    'threshold': 0.1
}

grass_color_predicate = ColorPredicate("grass", "../data/grass/images")
grass_color_predicate.create_multidimensional_histogram(**hist_args)
grass_color_predicate.create_gaussian_smoothed_histogram(**gauss_hist_args)
grass_color_predicate.plot_gaussian_smoothed_histogram(save=True)
