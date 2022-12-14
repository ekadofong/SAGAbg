# \\ computed from PyNeb with 
# \\ vac-to-air conversions from specutils.utils
balmer_wavelengths = {'Halpha':6563.,'Hbeta':4862., 'Hgamma':4340., 'Hdelta':4101. }
OIII_wavelengths = {'[OIII]5007': 5006.766373938368,
                    '[OIII]4959': 4958.83507790285,                    
                    '[OIII]4363': 4363.142139776673}
OII_wavelengths = {'[OII]3729':[3728.756593589433,3725.974257426484], #, '[OII]3727': 3725.974257426484,
                   '[OII]7320':[7320.,7318.9], '[OII]7330':[7329.7,7330.7]}
                   #'[OII]7320':7320.0, '[OII]7319':7318.9, '[OII]7330':7329.7, '[OII]7331':7330.7 }
NII_wavelengths = {'[NII]6548': 6547.952750702898,
                   '[NII]6583': 6583.354593822465}
                   #'[NII]5755': 5754.5071992902085}
SII_wavelengths = {'[SII]6731': 6730.713900332137, '[SII]6717': 6716.33869898957}
FeX = {}#'FeX':6374.}

line_ratios = [
               ("H", 1, 'Halpha', 'Hbeta'),
               #("H", 1, 'Halpha', 'Hdelta'),
               #("H", 1, 'Halpha', 'Hgamma'),
               ('S', 2, '[SII]6717', '[SII]6731'), 
               #('N',2, '[NII]5755', '[NII]6548'),              
               ('O',2,'[OII]7330', '[OII]3729'),
               ('O',2,'[OII]7320', '[OII]3729'),
               ('O',3,'[OIII]4363', '[OIII]5007')
               ]

# \\ testing to see if we make an improvement over theoretical limits with
# \\ resulting constraints on the observed line ratio
line_ratio_theorymax = { 
                        #0.493,  #\\ NII5755/6548
                        '[OII]7330':0.0456, #\\ OII7330/3729+3727
                        '[OII]7320':0.0564, #\\ OII7320/3729+3727
                        '[OIII]4363':0.160,  #\\ OIII4363/5007
}
#line_ratio_theorymin = [
#                        0.,
#                        0.685,
#                        0.,
#                        0.,
#                        0.
#                        ]

line_lists = [balmer_wavelengths, OIII_wavelengths, OII_wavelengths, SII_wavelengths, NII_wavelengths, FeX]
line_wavelengths = {}
for ll in line_lists:
    line_wavelengths.update(ll)
Nlines = len(line_wavelengths.keys())

#CONTINUUM_TAGS = ['Halpha','Hbeta','Hgamma','Hdelta','[OIII]5007','[SII]6717','FeX','[OII]3729','[OII]7320', '[NII]5755']
CONTINUUM_TAGS = ['Halpha','Hbeta','Hgamma','Hdelta','[OIII]5007','[SII]6717','[OII]3729','[OII]7320']
BALMER_ABSORPTION = ['Halpha','Hbeta','Hgamma','Hdelta']

## DEFAULT VALUES
DEFAULT_WINDOW_WIDTH = 140.
DEFAULT_LINE_WIDTH = 14.

## Lines to remove if they are not detected
## Since there is a lot of correlated noise in the area
remove_if_exceeds = ['[OII]7320', '[OII]7330']