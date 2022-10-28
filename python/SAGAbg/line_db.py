# \\ computed from PyNeb with 
# \\ vac-to-air conversions from specutils.utils
balmer_wavelengths = {'Halpha':6563.,'Hbeta':4862., 'Hgamma':4340., 'Hdelta':4101. }
OIII_wavelengths = {'[OIII]4959': 4958.83507790285,
                    '[OIII]5007': 5006.766373938368,
                    '[OIII]4363': 4363.142139776673}
OII_wavelengths = {}#'[OII]3729': 3728.756593589433, '[OII]3726': 3725.974257426484}
NII_wavelengths = {'[NII]6548': 6547.952750702898,
                   '[NII]6583': 6583.354593822465,}
                   #'[NII]5755': 5754.5071992902085}
SII_wavelengths = {'[SII]6731': 6730.713900332137, '[SII]6716': 6716.33869898957}
FeX = {'FeX':6374.}

line_lists = [balmer_wavelengths, OIII_wavelengths, OII_wavelengths, SII_wavelengths, NII_wavelengths, FeX]
line_wavelengths = {}
for ll in line_lists:
    line_wavelengths.update(ll)

CONTINUUM_TAGS = ['Halpha','Hbeta','Hgamma','Hdelta','[OIII]5007','[SII]6716','FeX','[OII]3729']
BALMER_ABSORPTION = ['Halpha','Hbeta','Hgamma','Hdelta']

## DEFAULT VALUES
DEFAULT_WINDOW_WIDTH = 140.
DEFAULT_LINE_WIDTH = 14.