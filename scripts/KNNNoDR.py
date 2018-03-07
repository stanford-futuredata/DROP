#!/usr/bin/env python
import subprocess

datasets = ['labeled_good_sinusoids_1e4_5e3','labeled_med_sinusoids_1e4_5e3','labeled_medbad_sinusoids_1e4_5e3',
            'labeled_bad_sinusoids_1e4_5e3']

datasets += ['wafer',
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

numTrials = 5


proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.KNNNoDRExperiments %s %d"
comp = "mvn compile"
package = "mvn package"
subprocess.call(comp, shell=True)
subprocess.call(package, shell=True)

for dataset in datasets:
    subprocess.call(proc % (dataset, numTrials), shell=True)

