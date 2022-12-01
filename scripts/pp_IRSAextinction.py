import catalogs


def build_IRSAcatalogs ( parent, chunk=10000, savedir='../local_data/galactic_extinction/' ):
    start = 0
    while start < parent.shape[0]:
        parent.iloc[start:(start+chunk)][['RA','DEC']].to_csv(f'{savedir}/input/IEXT_{start}_{start+chunk}.csv')
        start += chunk