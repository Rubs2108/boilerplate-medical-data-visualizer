import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv",
                header=0,
                sep=",",
                skipinitialspace=True
                )


# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
        var_name="variable",
        value_name="value"
    )
    


    # 6
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index=False).size().rename(columns={"size": "total"})
    
    

    # 7
    graph = sns.catplot(
        data=df_cat,
        x="variable", y="total",
        hue="value",
        col="cardio",
        kind="bar",
        height=4, aspect=1.2
    )
    graph.set_axis_labels("variable", "total")
    graph.set_titles("cardio = {col_name}" )
    for ax in graph.axes.flat:
        ax.tick_params(axis="x", labelrotation=45)


    # 8
    fig = graph.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    

    # 11
    he_low = df["height"].quantile(0.025)
    he_high = df["height"].quantile(0.975)
    we_low = df["weight"].quantile(0.025)
    we_high = df["weight"].quantile(0.975)

    mask_bp= df["ap_lo"] <= df["ap_hi"]
    mask_h= df["height"].between(he_low, he_high)
    mask_w= df["weight"].between(we_low, we_high)

    df_heat = df[mask_bp & mask_h & mask_w]


    # 12
    corr = df_heat.corr(numeric_only=True)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )
    ax.set_title("Heatmap")

    # 15



    # 16
    fig.savefig('heatmap.png')
    return fig
