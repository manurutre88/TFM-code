import pandas as pd
import tsfel

# ////////////////////////////////////////////////////////////////////////////////////////////

def ext_caract(df, sufix):
    mean, median, std, var = [], [], [], []

    for _, row in df.iterrows():
        mean.append(tsfel.feature_extraction.features.calc_mean(row))
        median.append(tsfel.feature_extraction.features.calc_median(row))
        std.append(tsfel.feature_extraction.features.calc_std(row))
        var.append(tsfel.feature_extraction.features.calc_var(row))

    ext_caract = pd.DataFrame({
        f'mean{sufix}': mean,
        f'median{sufix}': median,
        f'std{sufix}': std,
        f'var{sufix}': var
    })
    ext_caract.index = [f'User_{i+1}' for i in range(ext_caract.shape[0])]

    return ext_caract

# ////////////////////////////////////////////////////////////////////////////////////////////
