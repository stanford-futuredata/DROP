#!/usr/bin/env python
import subprocess
import sys



datasets = { "1":[],
             "2" :['ElectricDevices'],
             "3" :['FordA'],
             "4":[]
            }


lbrs = [0.99]
numTrials = 5


proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.AlterKNNWorkloadExperiments %s %f %d"
comp = "mvn compile"
package = "mvn package"
#subprocess.call(comp, shell=True)
#subprocess.call(package, shell=True)


for lbr in lbrs:
    for dataset in datasets[sys.argv[1]]:
        subprocess.call(proc % (dataset, lbr, numTrials), shell=True)

