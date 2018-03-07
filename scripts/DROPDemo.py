#!/usr/bin/env python
import subprocess

dataset = 'wafer'
lbr = 0.99

proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' demo.DROPDemo %s %f"
comp = "mvn compile"
package = "mvn package"
subprocess.call(comp, shell=True)
subprocess.call(package, shell=True)

subprocess.call(proc % (dataset, lbr), shell=True)

