# Evaluation metrics for model comparison

"""
    rmse(predicted, observed)

Root Mean Square Error.
"""
function rmse(predicted::AbstractVector, observed::AbstractVector)
    return sqrt(mean((predicted .- observed).^2))
end

"""
    correlation(predicted, observed)

Pearson correlation coefficient.
"""
function correlation(predicted::AbstractVector, observed::AbstractVector)
    return cor(predicted, observed)
end

"""
    skill_score(predicted, observed, reference)

Skill score relative to reference model:
  SS = 1 - MSE(predicted) / MSE(reference)
SS > 0 means predicted is better than reference.
"""
function skill_score(predicted::AbstractVector, observed::AbstractVector,
                     reference::AbstractVector)
    mse_pred = mean((predicted .- observed).^2)
    mse_ref = mean((reference .- observed).^2)
    return 1.0 - mse_pred / max(mse_ref, 1e-20)
end

"""
    prediction_efficiency(predicted, observed)

Nash-Sutcliffe prediction efficiency (PE):
  PE = 1 - Σ(pred - obs)² / Σ(obs - mean(obs))²
PE = 1 is perfect, PE = 0 is no better than mean, PE < 0 is worse.
"""
function prediction_efficiency(predicted::AbstractVector,
                               observed::AbstractVector)
    ss_res = sum((predicted .- observed).^2)
    ss_tot = sum((observed .- mean(observed)).^2)
    return 1.0 - ss_res / max(ss_tot, 1e-20)
end

"""
    metrics_summary(predicted, observed; name="Model")

Compute all metrics, return as NamedTuple.
"""
function metrics_summary(predicted::AbstractVector, observed::AbstractVector;
                         name::String="Model")
    return (
        name = name,
        rmse = rmse(predicted, observed),
        corr = correlation(predicted, observed),
        pe = prediction_efficiency(predicted, observed),
        n_points = length(predicted)
    )
end
