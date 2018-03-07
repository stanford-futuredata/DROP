#!/usr/bin/env python
import subprocess

datasets = ['wafer',
             'yoga',
             'uWaveGestureLibrary_Y',
             'uWaveGestureLibrary_X',
             'uWaveGestureLibrary_Z',
             'ElectricDevices',
             'Phoneme',
             'FordB',
             'MALLAT',
             'FordA',
             'NonInvasiveFatalECG_Thorax1',
             'NonInvasiveFatalECG_Thorax2',
             'UWaveGestureLibraryAll',
             'StarLightCurves']


datasets += ['labeled_good_sinusoids_1e4_5e3','labeled_med_sinusoids_1e4_5e3','labeled_medbad_sinusoids_1e4_5e3',
            'labeled_bad_sinusoids_1e4_5e3']

numTrials = 5
lbrs = [0.7]

proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.SVDKnnBaselineExperiment %s %f %d"
comp = "mvn compile"
package = "mvn package"
#subprocess.call(comp, shell=True)
#subprocess.call(package, shell=True)

for dataset in datasets:
    for lbr in lbrs:
        subprocess.call(proc % (dataset, lbr, numTrials), shell=True)

