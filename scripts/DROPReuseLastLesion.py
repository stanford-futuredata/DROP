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
lbrs = [0.70]
numTrials = 5

#datasets = ['StarLightCurves']
datasets = ['music_features','MNIST_all']

proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.DROPLesionReuseLastExperiments %s %f %d"
comp = "mvn compile"
package = "mvn package"
#subprocess.call(comp, shell=True)
#subprocess.call(package, shell=True)

for dataset in datasets:
    for lbr in lbrs:
        subprocess.call(proc % (dataset, lbr, numTrials), shell=True)

