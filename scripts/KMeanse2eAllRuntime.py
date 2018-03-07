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
ks = [2, 2, 8,8,8,7,39,2,8,2,42,42,8,3]

#datasets += ['labeled_good_sinusoids_1e4_5e3','labeled_med_sinusoids_1e4_5e3','labeled_medbad_sinusoids_1e4_5e3',
#            'labeled_bad_sinusoids_1e4_5e3']
#ks += [5,10,10,20]


#datasets += ['labeled_good_sinusoids_5e4_5e3','labeled_med_sinusoids_5e4_5e3','labeled_medbad_sinusoids_5e4_5e3',
#             'labeled_bad_sinusoids_5e4_5e3']
#ks += [5,10,10,20]
#datasets += ['labeled_good_sinusoids_1e5_5e3','labeled_med_sinusoids_1e5_5e3','labeled_medbad_sinusoids_1e5_5e3',
#             'labeled_bad_sinusoids_1e5_5e3']
#ks += [5,10,10,20]

lbrs = [0.99,.9,.8,.7]
numTrials = 5


proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.FullKmeansExperiments %s %f %d %d"
comp = "mvn compile"
package = "mvn package"
#subprocess.call(comp, shell=True)
#subprocess.call(package, shell=True)


for lbr in lbrs:
    for  k,dataset in zip(ks,datasets):
        subprocess.call(proc % (dataset, lbr,k, numTrials), shell=True)

