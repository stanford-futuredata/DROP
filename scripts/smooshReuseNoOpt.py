#!/usr/bin/env python
import subprocess

datasets = ["50words", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration", "CinC", "Coffee", "Computers", "Cricket", "Cricket", "Cricket", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FISH", "FordA", "FordB", "Gun", "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lighting2", "Lighting7", "MALLAT", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFatalECG", "NonInvasiveFatalECG", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface", "SonyAIBORobotSurfaceII", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols", "synthetic", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "Two", "UWaveGestureLibraryAll", "uWaveGestureLibrary", "uWaveGestureLibrary", "uWaveGestureLibrary", "wafer", "Wine", "WordsSynonyms", "Worms", "WormsTwoClass", "yoga", "MNIST_all"]


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
datasets = ["labeled_bad_sinusoids_1e4_1e3_trunc"]

lbrs = [0.70, 0.80, 0.90, 0.98]
opt = "NOOPTIMIZE"
q = 1.96


proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.SmooshReuseExperiments %s %f %f %s"
comp = "mvn compile"
package = "mvn package"
subprocess.call(comp, shell=True)
subprocess.call(package, shell=True)

for dataset in datasets:
    for lbr in lbrs:
        subprocess.call(proc % (dataset, lbr, q, opt), shell=True)


