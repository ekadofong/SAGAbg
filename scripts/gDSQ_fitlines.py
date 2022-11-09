import argparse

def build_dsqQ ( step, n_objects=0, source='SBAM', dropbox_directory=None ):
    if dropbox_directory is None:
        dropbox_directory = '../local_data/'
    if n_objects==0:
        import catalogs
        if source == 'SBAM':
            df = catalogs.build_SBAM(dropbox_directory=dropbox_directory)
        elif source == 'SBAMsats':
            df = catalogs.build_SBAM(dropbox_directory=dropbox_directory)
            df = df.query('p_sat_corrected==1')
            
        n_objects = df.shape[0]
    
    with open('./dsq_fitlines.txt', 'w') as f:
        for start_index in range(0, n_objects, step):
            end_index = min ( start_index + step, n_objects )
            #print(start_index, end_index )
            print ( "python do_bayesianfitlines.py -S %i -E %i -d %s --source SBAMsat --nfig 100" % (start_index, end_index, dropbox_directory), file=f)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser ( prog='gDSQ_fitlines.py', description='generate Dead Simple Queue worklist for fitlines' )
    parser.add_argument ( '--step', '-s', action='store', default=1)
    parser.add_argument ( '--nobjects', '-N', action='store', default=0)
    args = parser.parse_args ()
    step = int(args.step)
    build_dsqQ ( step, n_objects=int(args.nobjects) )
