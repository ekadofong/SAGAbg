import argparse


def build ( taskname, step, n_objects=0, source='SBAM', dropbox_directory=None, **kwargs ):
    # \\ read in from catalog if n_objects not supplied
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
    
    kwarg_string = ' '.join([ f'--{key} {val}' for key, val in kwargs.items() ])
    with open(f'./DSQ{taskname}.txt', 'w') as f:
        for start_index in range(0, n_objects, step):
            end_index = min ( start_index + step, n_objects )
            #print(start_index, end_index )            
            print ( 
                   f"python do_{taskname}.py -S %i -E %i -d %s %s" % (start_index, end_index, dropbox_directory, kwarg_string), 
                   file=f
                   )
        
def build_bayesianfitlines ( *args, **kwargs ):
    build ( 'bayesianfitlines', *args, **kwargs )

def dep_build_dsqQ ( step, n_objects=0, source='SBAM', dropbox_directory=None ):
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
            print ( "python do_bayesianfitlines.py -S %i -E %i -d %s --source SBAMsat --nfig 100" %\
                        (start_index, end_index, dropbox_directory), file=f)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser ( prog='gDSQ.py', description='generate Dead Simple Queue worklist for fitlines' )
    parser.add_argument ( '--step', '-s', action='store', default=10)
    parser.add_argument ( '--nobjects', '-N', action='store', default=0)
    parser.add_argument ( '--taskname', action='store', default='bayesianfitlines'  )
    parser.add_argument ( '--source', '-S', action='store', default='SBAM' )
    #parser.add_argument ( '--nfig', action='store', default=10 )
    args = parser.parse_args ()
    #step = int(args.step)
    build ( args.taskname, int(args.step), n_objects=int(args.nobjects), source=args.source )
    #build_dsqQ ( step, n_objects=int(args.nobjects) )
