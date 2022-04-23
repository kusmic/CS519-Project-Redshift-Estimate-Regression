Each separate source code file was run differently depending on the
user who created it.

eda, pca, random_forest_regressor, astro_process, Nonlinear_Regression: 
This was run on a native machine. In general they need scikit-learn v.0.24.2 
and numpy v.1.20.3, and astro_process needs astropy v.4.3.1 to run properly. 
They are meant to run on command line after some slight augmentation, e.g. 
python <filename.py>

dNN: This is a Jupyter notebook and was originally run on a native machine,
but can be transferred to a Google Colab environment. If one does use Colab,
you need to include the very first line mounting your drive folder (i.e.
from google.colab import drive; drive.mount("/content/drive") ), then
rename the the paths to the absolute path into your working Drive directory.
If using native machine, above dependencies are necessary, except astropy, and
pytorch v.1.11.0 to run.
