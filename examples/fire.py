from src.colorpredicate import ColorPredicate

hist_args = {
    'color_space': 'bgr',
    'ch_indexes': (0, 1, 2),
    'bins': (16, 16, 16),
    'target_sr': 1.0,
    'bg_rate': 1.0
}

gauss_hist_args = {
    't_amp': 1.0,
    't_cov': 0.05,
    'bg_amp': 0.5,
    'bg_cov': 0.025,
    'threshold': 0.01
}

fire_color_predicate = ColorPredicate("fire", "../data/fire/images")
fire_color_predicate.load_masks("../data/fire/ground_truth")
fire_color_predicate.create_multidimensional_histogram(**hist_args)
fire_color_predicate.create_gaussian_smoothed_histogram(**gauss_hist_args)
fire_color_predicate.plot_gaussian_smoothed_histogram(save=True)
