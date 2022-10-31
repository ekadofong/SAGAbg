import argparse

def build_dsqQ ( step, nobj=0, source='SBAM', dropbox_directory=None ):
    if dropbox_directory is None:
        dropbox_directory = '../local_data/'
    if n_objects==0:
        import catalogs
        if source == 'SBAM':
            df = catalogs.build_SBAM(dropbox_directory=dropbox_directory)
        n_objects = df.shape[0]
    
    with open('./dsq_fitlines.txt', 'w') as f:
        for start_index in range(0, n_objects, step):
            end_index = min ( start_index + step, n_objects )
            #print(start_index, end_index )
            print ( "module load miniconda; conda activate vgrace; python do_fitlines.py -S %i -E %i -d %s" % (start_index, end_index,
                                                                                                               dropbox_directory),
                    file=f)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser ( prog='gDSQ_fitlines.py', description='generate Dead Simple Queue worklist for fitlines' )
    parser.add_argument ( '--step', '-s', action='store', default=1)
    args = parser.parse_args ()
    step = int(args.step)
    build_dsqQ ( step )
