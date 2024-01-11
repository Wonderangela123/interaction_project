import pandas as pd

df = pd.read_csv('/work/aliu10/interaction_project/GBMLGG/firebrowse/labels.csv', sep=' ')
df = df.transpose()
df.index = ['-'.join(idx.split('-')[:4]) for idx in df.index]
df.index = [idx[:-1].upper() for idx in df.index]
df.iloc[:,0] = df.iloc[:,0].replace({'gbm': 0, 
                                     'lgg': 1})

mrna = pd.read_csv('/work/aliu10/interaction_project/GBMLGG/firebrowse/gdac.broadinstitute.org_GBMLGG.mRNAseq_Preprocess.Level_3.2016012800.0.0/GBMLGG.uncv2.mRNAseq_RSEM_all.txt', sep='\t', index_col=0)
mrna = mrna.transpose()

# mirna = pd.read_csv('/work/aliu10/interaction_project/GBMLGG/firebrowse/gdac.broadinstitute.org_GBMLGG.miRseq_Preprocess.Level_3.2016012800.0.0/GBMLGG.miRseq_RPKM.txt', sep='\t', index_col=0)
# mirna = mirna.transpose()
# # mirna contains lgg only

methy = pd.read_csv('/work/aliu10/interaction_project/GBMLGG/firebrowse/gdac.broadinstitute.org_GBMLGG.Methylation_Preprocess.Level_3.2016012800.0.0/GBMLGG.meth.by_mean.data.txt', sep='\t', index_col=0)
methy = methy.transpose()
methy.drop(methy.columns[0], axis=1, inplace=True)
methy.fillna(0, inplace=True)

common_indices = df.index.intersection(mrna.index).intersection(methy.index)
df_common = df.loc[common_indices]
mrna_common = mrna.loc[common_indices]
# mirna_common = mirna.loc[common_indices]
methy_common = methy.loc[common_indices]

for i in range(1,6):
    # split training and testing sets
    tr_1 = df_common.loc[df_common.iloc[:,0] == 0,:].sample(frac=0.7)
    te_1 = df_common.loc[df_common.iloc[:,0] == 0,:].drop(tr_1.index)

    tr_2 = df_common.loc[df_common.iloc[:,0] == 1,:].sample(frac=0.7)
    te_2 = df_common.loc[df_common.iloc[:,0] == 1,:].drop(tr_2.index)

    df_common_tr = pd.concat([tr_1, tr_2])
    df_common_te = pd.concat([te_1, te_2])
    df_common_tr.loc[:, 0].to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/labels_tr.csv".format(i), sep=",", index=False, header=False)
    df_common_te.loc[:, 0].to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/labels_te.csv".format(i), sep=",", index=False, header=False)

    mrna_features_tr = mrna.loc[df_common_tr.index,:]
    mrna_features_te = mrna.loc[df_common_te.index,:]
    mrna_features_tr.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/1_tr.csv".format(i), sep=",", index=False, header=False)
    mrna_features_te.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/1_te.csv".format(i), sep=",", index=False, header=False)
    features = pd.DataFrame(mrna_features_te.columns)
    # remove gene symbols containg '?'
    features.iloc[:,0] = features.iloc[:,0].str.split('|').str[0]
    features = features[~features.iloc[:,0].str.contains('\?')]
    features.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/1_featname.csv".format(i),  index=False, header=False)

    # mirna_features_tr = mirna.loc[df_common_tr.index,:]
    # mirna_features_te = mirna.loc[df_common_te.index,:]
    # mirna_features_tr.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/2_tr.csv".format(i), sep=",", index=False, header=False)
    # mirna_features_te.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/2_te.csv".format(i), sep=",", index=False, header=False)
    # features = pd.DataFrame(mirna_features_te.columns)
    # features.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/2_featname.csv".format(i),  index=False, header=False)

    methy_features_tr = methy.loc[df_common_tr.index,:]
    methy_features_te = methy.loc[df_common_te.index,:]
    methy_features_tr.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/2_tr.csv".format(i), sep=",", index=False, header=False)
    methy_features_te.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/2_te.csv".format(i), sep=",", index=False, header=False)
    features = pd.DataFrame(methy_features_te.columns)
    features.to_csv("/work/aliu10/interaction_project/GBMLGG/processed/{}/2_featname.csv".format(i),  index=False, header=False)