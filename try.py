import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA

transformer = PowerTransformer(method='yeo-johnson', standardize=True)

raw_data = pd.read_csv('./train.csv')
raw_data = raw_data.drop(columns=['defects', 'id'])
pipeline = make_pipeline(

    QuantileTransformer(output_distribution = 'normal',random_state = 42),
# PowerTransformer(method='yeo-johnson', standardize=True),
    StandardScaler()
)
transformer = make_column_transformer(
    (
        pipeline,
        make_column_selector(dtype_include=np.number) # We want to apply numerical_pipeline only on numerical columns
    ),
    remainder='passthrough',
    verbose_feature_names_out=False
)
normalized_data = pipeline.fit_transform(raw_data)
normalized_df = pd.DataFrame(data=normalized_data, columns=raw_data.columns)

# mean = normalized_df.mean(axis=0)
pca = PCA()
pca.fit(normalized_df)
transformed_data = pca.transform(normalized_df)





#,,,n,v,l,d,i,e,b,t,lOCode,lOComment,lOBlank,locCodeAndComment,uniq_Op,uniq_Opnd,total_Op,total_Opnd,branchCount
#plt.hist(normalized_df['loc'], bins=200, edgecolor='k', alpha=0.7, density=True)
#plt.hist(normalized_df['v(g)'], bins=300, density=True, edgecolor='k', alpha=0.7)
#plt.hist(normalized_df['ev(g)'], bins=200, edgecolor='k', alpha=0.7, density=True)
#plt.hist(normalized_df['iv(g)'], bins=200, edgecolor='k', alpha=0.7, density=True)
#plt.hist(normalized_df['n'], bins=200, edgecolor='k', alpha=0.7, density=True)
#plt.hist(normalized_df['v'], bins=200, edgecolor='k', alpha=0.7, density=True)
#plt.hist(normalized_df['l'], bins=200, edgecolor='k', alpha=0.7, density=True)
#plt.hist(normalized_df['d'], bins=200, edgecolor='k', alpha=0.7, density=True)
#plt.hist(normalized_df['i'], bins=200, edgecolor='k', alpha=0.7, density=True)
#plt.hist(normalized_df['e'], bins=50, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['b'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['t'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['lOCode'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['lOComment'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['lOBlank'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['locCodeAndComment'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['uniq_Op'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['uniq_Opnd'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['total_Op'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['total_Opnd'], bins=200, edgecolor='k', alpha=0.7, density=True)
# plt.hist(normalized_df['branchCount'], bins=200, edgecolor='k', alpha=0.7, density=True)

# num = normalized_df.columns
#
# df = normalized_df
#
# # Use of more advanced artistic matplotlib interface (see the axes)
# fig, axes = plt.subplots(len(num), 2 ,figsize = (16, len(num) * 4), gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20]})
#
# for i,col in enumerate(num):
#     ax = axes[i,0]
#     sns.kdeplot(data = pd.DataFrame(df[col]), linewidth = 2.1, warn_singular=False, ax = ax) # Use of seaborn with artistic interface
#     ax.set_title('train', fontsize = 9)
#     ax.grid(visible=True, linestyle = '--', color='lightgrey', linewidth = 0.75)
#     ax.set(xlabel = '', ylabel = '')
#     ax.set_xlim(-3, 3)
#
#     ax = axes[i,1]
#     sns.boxplot(data = pd.DataFrame(df[col]), y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#456cf0', ax = ax)
#     ax.set(xlabel = '', ylabel = '')
#     ax.set_title("Train", fontsize = 9)



# plt.tight_layout()
# plt.show()


# normalized_df.describe().T\
#     .style.bar(subset=['mean'], color=px.colors.qualitative.G10[2])\
#     .background_gradient(subset=['std'], cmap='Blues')\
#     .background_gradient(subset=['50%'], cmap='Reds')


    #raw_data[i] = stats.boxcox(transformed_data, lmbda=lambda_best_fit, inverse=True)
# data = pd.concat([raw_data.drop(columns=['defects']), raw_data['defects']], axis=0)

# metrics = raw_data.describe()
# for i in raw_data.columns:
#     if i != "defects":
#         Q1 = metrics[i]["25%"]
#         Q3 = metrics[i]["75%"]
#         IQR = Q3 - Q1
#         upper = Q3 + IQR * 3
#         lower = Q1 - IQR * 3
#         raw_data[i][raw_data[i] > upper] = np.NaN
#         raw_data[i][raw_data[i] < lower] = np.NaN
#
# raw_data.dropna(inplace=True)
# raw_data.reset_index(drop=True, inplace=True)
# A = pd.DataFrame([1,2,3])
# B = pd.DataFrame([4,5,6])
# C = pd.concat([A, B], axis=1)
print(raw_data)

