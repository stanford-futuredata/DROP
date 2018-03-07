#!/usr/bin/env python
import subprocess

datasets = ['wafer',
            #'InlineSkate', m < n
            'yoga',
            'uWaveGestureLibrary_Y',
            'uWaveGestureLibrary_X',
            'uWaveGestureLibrary_Z',
            'ElectricDevices',
            'Phoneme',
            'FordB',
            #'CinC_ECG_torso', m < n
            'MALLAT',
            'FordA',
            'NonInvasiveFatalECG_Thorax1',
            'NonInvasiveFatalECG_Thorax2',
            #'HandOutlines', m < n
            'UWaveGestureLibraryAll',
            'StarLightCurves']
datasets += ["bad", "medium", "good", "bad_big", "medium_big", "good_big"]

datasets = ['labeled_bad_sinusoids_1e4_1e3_trunc']
lbrs = [0.7, 0.8, 0.9, 0.98]
optimize = "OPTIMIZE"
numTrials = 5

#datasets = ['StarLightCurves']

proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.DROPKNNWrongCostExperiments %s %f %s %d"
comp = "mvn compile"
package = "mvn package"
subprocess.call(comp, shell=True)
subprocess.call(package, shell=True)

for dataset in datasets:
    for lbr in lbrs:
        subprocess.call(proc % (dataset, lbr, optimize, numTrials), shell=True)

