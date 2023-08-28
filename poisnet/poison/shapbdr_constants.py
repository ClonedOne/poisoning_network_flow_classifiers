"""
Constant values for the SHAP-based poisoning attack
"""

# Criteria for the attack strategies
feature_selection_criterion_snz = 'shap_nearest_zero_nz'
feature_selection_criterion_sna = 'shap_nearest_zero_nz_abs'
feature_selection_criterion_mip = 'most_important'
feature_selection_criterion_fix = 'fixed'
feature_selection_criterion_large_shap = 'shap_largest_abs'
feature_selection_criterion_fshap = 'fixed_shap_nearest_zero_nz_abs'
feature_selection_criterion_combined = 'combined_shap'
feature_selection_criterion_combined_additive = 'combined_additive_shap'
feature_selection_criterion_combined_lowerbound = 'combined_lowerbound_shap'
feature_selection_criteria = {
    feature_selection_criterion_snz,
    feature_selection_criterion_sna,
    feature_selection_criterion_mip,
    feature_selection_criterion_fix,
    feature_selection_criterion_large_shap,
    feature_selection_criterion_fshap,
    feature_selection_criterion_combined,
    feature_selection_criterion_combined_additive,
    feature_selection_criterion_combined_lowerbound,
}

value_selection_criterion_min = 'min_population_new'
value_selection_criterion_shap = 'argmin_Nv_sum_abs_shap'
value_selection_criterion_combined = 'combined_shap'
value_selection_criterion_combined_additive = 'combined_additive_shap'
value_selection_criterion_combined_lowerbound = 'combined_lowerbound_shap'
value_selection_criterion_fix = 'fixed'
value_selection_criteria = {
    value_selection_criterion_min,
    value_selection_criterion_shap,
    value_selection_criterion_combined,
    value_selection_criterion_fix,
    value_selection_criterion_combined_additive,
    value_selection_criterion_combined_lowerbound
}
