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


ks = [0.98584873545531315,
 2.4716254615961679,
 5.6448885640304116,
 6.1434837864217844,
 5.4457551731808351,
 6.5982789124660641,
 35.607970609258835,
 20.898696045117966,
 2.9831531198516981,
 20.423485141103111,
 2.1870417603321606,
 2.0491512956289757,
 14.493321327908308,
 4.1964362301155536]

ks = [46.657517144170285,
 62.74899864787713,
 60.989140908516219,
 65.372820751160148,
 75.17908788257219,
 37.294635807205545,
 134.55154425630514,
 85.071398839938823,
 27.422147740500531,
 84.81728811663055,
 64.644947182628187,
 65.711096013876229,
 105.55553730997963,
 88.971354296714821]

ks = [0.98584873545531315,
 2.4716254615961679,
 5.6448885640304116,
 6.1434837864217844,
 5.4457551731808351,
 6.5982789124660641,
 35.607970609258835,
 20.898696045117966,
 2.9831531198516981,
 20.423485141103111,
 2.1870417603321606,
 2.0491512956289757,
 14.493321327908308,
 4.1964362301155536]


lbrs = [0.99,.9,.8,.7]
numTrials = 5


proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.FullDBSCANExperiments %s %f %f %d"
comp = "mvn compile"
package = "mvn package"
#subprocess.call(comp, shell=True)
#subprocess.call(package, shell=True)

for lbr in lbrs:
    for  k,dataset in zip(ks,datasets):
        subprocess.call(proc % (dataset, lbr,k, numTrials), shell=True)

